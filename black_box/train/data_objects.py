import json, torch, random
import numpy as np
from common.utils import load_wav
from eval.spectral_compare import estimate_noise_profile, spectral_subtract
from common.delay_ops import measure_delay, apply_shift

def parse_to_subarrays(arr, group_size):
    return [arr[i:i+group_size] for i in range(0, len(arr), group_size)] # slicing past the end of a list just returns whatever's left rather than erroring

class Chunk():
    def __init__(self, dry_data, wet_data=None, params={}):
        self.dry_data=dry_data
        self.wet_data=wet_data
        self.params=params
    
    def normalize_params(self, param_configs):
        norm_params={}
        for k, v in self.params.items():
            param_def = param_configs[k]
            if param_def['dtype'] == torch.float32:
                norm_params[k] = 2 * (v - param_def['min']) / (param_def['max'] - param_def['min']) - 1
        return norm_params
    
    def get_param_tensors(self, device, param_names, param_configs):
        norm_params = self.normalize_params(param_configs)
        return [torch.tensor([norm_params[pn]], dtype=param_configs[pn]['dtype'], device=device) for pn in param_names]
    
    def get_dry_tensor(self, device, gain=1):
        return torch.from_numpy(self.dry_data/gain).unsqueeze(0).to(device)

    def get_wet_tensor(self, device):
        if isinstance(self.wet_data, np.ndarray):
            return torch.from_numpy(self.wet_data).unsqueeze(0).to(device) 
        else: return None

    def get_target_tensor(self, device):
        if isinstance(self.wet_data, np.ndarray):
            return self.get_wet_tensor(device).unsqueeze(-1)
        else: return None

    def get_features_tensor(self, device, param_names, param_configs, gain=1):
        dry_tensor = self.get_dry_tensor(device, gain)
        # Make dry input (batch, seq_len, 1)
        batch, seq_len = dry_tensor.shape
        dry_view = dry_tensor.unsqueeze(-1)

        # Make param input (batch, seq_len, 1), same value repeated across time
        param_views = [p.view(batch, 1, 1).expand(batch, seq_len, 1) for p in self.get_param_tensors(device, param_names, param_configs)]
        return torch.cat([dry_view] + param_views, dim=-1)


class Segment():
    def __init__(self, chunks, noise_profile=None, wet_gain=None):
        self.chunks:list[Chunk]=chunks
        self.noise_profile=noise_profile
        self.wet_gain=wet_gain
    
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]
    
    def append(self, chunk):
        self.chunks.append(chunk)

    def __iter__(self):
        return iter(self.chunks)
    
    def get_features_tensors(self, device, param_names, param_configs, apply_gain=False):
        if apply_gain:
            assert self.wet_gain, 'Segment must have wet_gain value to apply wet gain to chunk'
        gain = self.wet_gain if apply_gain else 1
        return torch.cat([chunk.get_features_tensor(device, param_names, param_configs, gain=gain) for chunk in self.chunks], dim=1)

    def get_target_tensors(self, device):
        target_tensor = [chunk.get_target_tensor(device) for chunk in self.chunks]
        if None in target_tensor: return None
        return torch.cat(target_tensor, dim=1)


class Batch():
    def __init__(self, segments, tag='series'): # tag is a debugging term to tell if the batch is a timeseries of segments or a time windows of segments across tracks
        self.segments:list[Segment]=segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]

    def append(self, segment):
        self.segments.append(segment)

    def __iter__(self):
        return iter(self.segments)

    def shuffle(self, seed=42):
        random.seed(seed)
        random.shuffle(self.segments)

    def get_tensors(self, device, param_names, param_configs, apply_gain=False):
        features_tensors = torch.cat([segment.get_features_tensors(device, param_names, param_configs, apply_gain=apply_gain) for segment in self.segments], dim=0)
        tgt=[segment.get_target_tensors(device) for segment in self.segments]
        target_tensors = torch.cat(tgt, dim=0) if not None in tgt else None
        gains_tensor = torch.tensor([segment.wet_gain for segment in self.segments], device=device, dtype=torch.float32) if apply_gain else None # (batch_size,)
        return [features_tensors, target_tensors, gains_tensor]


class Track():
    def __init__(self, segments):
        self.segments:list[Segment]=segments

    def shuffle(self, seed=42):
        random.seed(seed)
        random.shuffle(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]

    def append(self, segment):
        self.segments.append(segment)

    def __iter__(self):
        return iter(self.segments)
    
    def get_dry(self):
        chunks_data = [chunk.dry_data for segment in self.segments for chunk in segment]
        return np.concatenate(chunks_data)
        # data=[]
        # for segment in self.segments:
        #     for chunk in segment:
        #         data += chunk.dry_data
        
    def get_wet(self):
        chunks_data = [chunk.wet_data for segment in self.segments for chunk in segment]
        return np.concatenate(chunks_data)
        # data=[]
        # for segment in self.segments:
        #     for chunk in segment:
        #         data += chunk.wet_data

    def compute_wet_gain(self):
        rms_d = np.sqrt(np.mean(self.get_dry() ** 2))
        rms_w = np.sqrt(np.mean(self.get_wet() ** 2))
        return rms_w / rms_d

    def get_noise_profile(self):
        return self.segments[0].noise_profile


class DataSet():
    def __init__(self, manifest_file, dry_file, wet_dir, chunk_seconds, param_names, param_configs, segment_size=20, silent_lead_in_seconds=8, trim_noise = True):
        self.tracks = []
        self.n_segments = 0
        self.chunk_seconds = chunk_seconds
        self.param_names = param_names
        self.param_configs = param_configs
        self.segment_size = segment_size
        self.silent_lead_in_seconds = silent_lead_in_seconds
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        track_segments = []

        # Get dry data
        print(f"Load dry: {dry_file}")
        dry_full, sr = load_wav(dry_file)
        n_trim = int(silent_lead_in_seconds * sr)
        dry_trim = dry_full[n_trim:]
        chunk_len = int(chunk_seconds * sr)

        # Get wet data
        with open(manifest_file, 'r') as f:
            man_records = [json.loads(line) for line in f if line.strip()]
        for man_record in man_records:
            wet_file = wet_dir + '/' + man_record['id'] + '.wav'
            print(f"Load wet: {wet_file}")
            wet_full, wet_sr = load_wav(wet_file)

            # Get params
            params=man_record['params']

            # Get noise
            if trim_noise:
                noise_profile = estimate_noise_profile(wet_full[:n_trim], wet_sr) # pre-silence
                wet_full = spectral_subtract(wet_full, noise_profile, sr)
            else: noise_profile=None
            assert wet_sr == sr, f"{wet_file} has a different sample rate than {dry_file}"
            wet_trim = wet_full[n_trim:]

            # Align dry and wet
            delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
            dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
            n_samples = len(dry_aligned)
            n_chunks = n_samples // chunk_len # number of full chunks in data
            # print(f"n_samples {n_samples} w n_chunks {n_chunks}")

            # Make chunks for this track
            chunk_bucket=[]
            segments=[]
            for i in range(n_chunks):
                s = i * chunk_len

                chunk = Chunk(dry_aligned[s:s + chunk_len].copy(), wet_aligned[s:s + chunk_len].copy(), params)

                chunk_bucket.append(chunk)
                if (i+1) % segment_size == 0:
                    segments.append(Segment(chunk_bucket, noise_profile))
                    chunk_bucket = []
            
            self.tracks.append(Track(segments))
            del dry_aligned, wet_aligned, wet_full   # explicitly drop the full-track arrays now that chunking is done
        
        # resize tracks to equal length
        self.n_segments = min([len(t.segments) for t in self.tracks])
        self.tracks = [Track(track.segments[:self.n_segments]) for track in self.tracks]

    def make_window_batches(self, track_size:int=30): # group for each trackset
        # track_size = batch_size
        # T_0 [  ][  ][  ]
        # T_1 [t0][t1][t2]
        # T_2 [  ][  ][  ]
        #        +
        # T_3 [  ][  ][  ]
        # T_4 [t0][t1][t2]
        # T_4 [  ][  ][  ]
        
        batch_groups=[]
        for track_group in parse_to_subarrays(self.tracks, track_size):
            batches=[]
            for s_i in range(0, self.n_segments):
                batches.append(Batch([track[s_i] for track in track_group], tag='window'))
            batch_groups.append(batches)
        return batch_groups


    def batches_of_random(self, batch_size=30, seed=42):

        # Flatten segments
        segments = []
        for track in self.tracks:
            for segment in track:
                segments.append(segment)

        # randomize segments
        random.seed(seed)
        random.shuffle(segments)

        # batch segments
        batches = []
        for batch_segments in parse_to_subarrays(segments, batch_size):
            batches.append(Batch(batch_segments, tag='random'))
        return batches
    
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]

    def append(self, track):
        self.tracks.append(track)

    def __iter__(self):
        return iter(self.tracks)

    def apply_gain_model(self, gain_model):
        for i, track in enumerate(self.tracks):
            print('BEFORE')
            for segment in track:
                print(f'{i}: {segment.wet_gain}')
        for track in self.tracks:
            for segment in track:
                segment.wet_gain = gain_model.predict(track)
        for i, track in enumerate(self.tracks):
            print('AFTER')
            for segment in track:
                print(f'{i}: {segment.wet_gain}')
##########################################################################################################################################
##########################################################################################################################################

# class PredictionSet():
#     def __init__(
#         predict_file=None,
#         predict_data=None, params=None, sr=None,
#         predict_manifest_record=None, predict_dir=None,
#         chunk_seconds=0.03,
#         silent_lead_in_seconds=8,
#         trim_noise=True
#     ):
#         if predict_file and params:
#             predict_data, sr = load_wav(predict_file)
#             # use supplied params
#         elif predict_data and params and sr:
#             pass
#         elif predict_manifest_record and predict_dir:
#             predict_data, sr = load_wav(predict_dir + '/' + predict_manifest_record['id'] + '.wav')
#             params=man_record['params']
#         else:
#             assert False, 'invalid predict params'


#             n_trim = int(silent_lead_in_seconds * sr)


#         if trim_noise:
#             noise_profile = estimate_noise_profile(predict_data[:int(silent_lead_in_seconds * sr)], sr) # pre-silence
#             predict_data = spectral_subtract(predict_data, noise_profile, sr)
#         else: noise_profile=None

#         # Get noise
#         if trim_noise:
#             predict_data = spectral_subtract(predict_data, noise_profile, sr)
#         chunk_len = int(chunk_seconds * sr)
#         n_chunks = len(predict_data) // chunk_len # number of full chunks in data


#         for i in range(n_chunks):
#             s = i * chunk_len

#             chunk = Chunk(dry_aligned[s:s + chunk_len].copy(), wet_aligned[s:s + chunk_len].copy(), params)

#             chunk_bucket.append(chunk)
#             if (i+1) % segment_length == 0:
#                 segments.append(Segment(chunk_bucket, noise_profile))
#                 chunk_bucket = []
        
#         self.track = Track(segments)
    
#     def make_window_batches(self): # group for each trackset
#         track_group=[self.track]

#         batches=[]
#         for s_i in range(0, self.n_segments):
#             batches.append(Batch([track[s_i]], tag='window'))
#         return batches