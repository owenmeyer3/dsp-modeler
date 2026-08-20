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

    def __len__(self):
        return len(self.dry_data)
    
    def normalize_params(self, param_configs:dict) -> dict: #{'d': -0.33333333333333337, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
        norm_params={}
        for k, v in self.params.items():
            param_def = param_configs[k]
            if param_def['dtype'] == torch.float32:
                norm_params[k] = 2 * (v - param_def['min']) / (param_def['max'] - param_def['min']) - 1
        return norm_params
    
    def get_param_tensors(self, device, param_names, param_configs)->list[torch.tensor]: # [tensor([-0.3333], device='cuda:0'), tensor([-0.3333], device='cuda:0'), tensor([-0.3333], device='cuda:0')]
        norm_params = self.normalize_params(param_configs)
        return [torch.tensor([norm_params[pn]], dtype=param_configs[pn]['dtype'], device=device) for pn in param_names]
    
    def get_dry_tensor(self, device) -> torch.tensor: # tensor([-0.0632, 0.0642, 0.0651, ...]
        return torch.from_numpy(self.dry_data).to(device)

    def get_wet_tensor(self, device) -> torch.tensor: # tensor([-0.0632, 0.0642, 0.0651, ...]
        if isinstance(self.wet_data, np.ndarray):
            return torch.from_numpy(self.wet_data).to(device) 
        else: return None

    def get_target_tensor(self, device)->list[torch.tensor]: # tensor(
                                                                # [-0.0632],
                                                                # [ 0.0642],
                                                                # [ 0.0651],
                                                            # )    
        if isinstance(self.wet_data, np.ndarray):
            return self.get_wet_tensor(device).unsqueeze(-1)
        else: return None

    def get_features_tensor(self, device, param_names, param_configs)-> torch.tensor: # tensor(
                                                                                            # [-0.0632, -0.3333, -0.3333, ...],
        # get dry tensor                                                                    # [ 0.0642, -0.3333, -0.3333, ...],
        dry_tensor = self.get_dry_tensor(device)                                            # [ 0.0651, -0.3333, -0.3333, ...],
                                                                                            # ...
        # stretch param tensors for each sample in chunk                                )
        param_tensors=self.get_param_tensors(device, param_names, param_configs)
        expanded_params = [param_tensor.repeat(len(dry_tensor)) for param_tensor in param_tensors]

        # combine to columns, transpose to time-space
        features_tensors=[dry_tensor]+expanded_params
        return torch.stack(features_tensors).T

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
    
    def get_features_tensor(self, device, param_names, param_configs): # tensor(
                                                                            # [-0.0632, -0.3333, -0.3333, ...],
                                                                            # [0.0642,  -0.3333, -0.3333, ...],
                                                                            # [0.0651,  -0.3333, -0.3333, ...],
                                                                        # )                               )
        return torch.cat([chunk.get_features_tensor(device, param_names, param_configs) for chunk in self.chunks], dim=0)

    def get_target_tensor(self, device):   # tensor(
                                                # [-0.0632],
                                                # [ 0.0642],
                                                # [ 0.0651],
                                            # )  
        return torch.cat([chunk.get_target_tensor(device) for chunk in self.chunks], dim=0)


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


    def get_features_tensor(self, device, param_names, param_configs): # tensor(
                                                                            # [-0.0632, -0.3333, -0.3333, ...],
                                                                            # [0.0642,  -0.3333, -0.3333, ...],
                                                                            # [0.0651,  -0.3333, -0.3333, ...],
                                                                        # )        
                                                                        # 
        segment_tensors = [s.get_features_tensor(device, param_names, param_configs) for s in self.segments]
        return torch.stack(segment_tensors, dim=0) 


    def get_target_tensor(self, device):   # tensor(
                                                # [-0.0632],
                                                # [ 0.0642],
                                                # [ 0.0651],
                                            # )  
        segment_tensors = [s.get_target_tensor(device) for s in self.segments]
        return torch.stack(segment_tensors, dim=0) 


    def get_gains_tensor(self, device, param_names):   # tensor(
                                                    # [0.5, 1, 1, ...],
                                                    # [0.5, 1, 1, ...],
                                                    # [0.5, 1, 1, ...],
                                                # )  
        segment_tensors=[]
        for segment in self.segments:
            gain_row = [segment.wet_gain]
            for i in param_names:
                gain_row.append(1.0)

            row_tensor = torch.tensor(gain_row, device=device, dtype=torch.float32)

            samples_per_seg = len(segment)*len(segment[0])
            segment_tensor = torch.stack([row_tensor] * samples_per_seg)

            segment_tensors.append(segment_tensor)

        return torch.stack(segment_tensors, dim=0)


    def get_tensors(self, device, param_names, param_configs):
        return self.get_features_tensor(device, param_names, param_configs), self.get_target_tensor(device)


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
        
    def get_wet(self):
        chunks_data = [chunk.wet_data for segment in self.segments for chunk in segment]
        return np.concatenate(chunks_data)

    def compute_wet_gain(self):
        rms_d = np.sqrt(np.mean(self.get_dry() ** 2))
        rms_w = np.sqrt(np.mean(self.get_wet() ** 2))
        return rms_w / rms_d

    def get_noise_profile(self):
        return self.segments[0].noise_profile


class DataSet():
    def __init__(self, manifest_file, dry_file, wet_dir, chunk_seconds, param_names, param_configs, segment_size=20, silent_lead_in_seconds=8, trim_noise = True):
        self.tracks:list[Track] = []
        self.sample_rate=None
        self.chunk_seconds = chunk_seconds
        self.param_names = param_names
        self.param_configs = param_configs
        self.segment_size = segment_size
        self.silent_lead_in_seconds = silent_lead_in_seconds
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        track_segments = []
        print("loading dataset")

        # Get dry data
        print(f"Load dry: {dry_file}")
        dry_full, sr = load_wav(dry_file)
        self.sample_rate = sr
        n_trim = int(silent_lead_in_seconds * sr)
        dry_trim = dry_full[n_trim:]
        chunk_len = int(chunk_seconds * sr)
        print(f"chunk_len {chunk_len}")
        print(f"segment_size {segment_size}")

        # Get wet data
        with open(manifest_file, 'r') as f:
            man_records = [json.loads(line) for line in f if line.strip()]
        for i, man_record in enumerate(man_records):
            wet_file = wet_dir + '/' + man_record['id'] + '.wav'
            print(f"Load wet ({i}/{len(man_records)}): {wet_file}")
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

            dry_chunks_data = parse_to_subarrays(dry_aligned, chunk_len)
            wet_chunks_data= parse_to_subarrays(wet_aligned, chunk_len)
            chunks = [Chunk(dry_chunks_data[i].copy(), wet_chunks_data[i].copy(), params) for i in range(len(dry_chunks_data))]

            segments_data = parse_to_subarrays(chunks, segment_size)
            segments = [Segment(sd, noise_profile) for sd in segments_data]
            self.tracks.append(Track(segments))

            del dry_aligned, wet_aligned, wet_full   # explicitly drop the full-track arrays now that chunking is done

        print(f"# Chunks {len(chunks)}")
        print(f"# Segments {len(segments)}")
        print(f"# Tracks {len(self.tracks)}")

        # resize tracks to equal length
        min_segments = min([len(track) for track in self.tracks]) - 1 # -1 cuts off any partial segments

        self.tracks = [Track(t[:min_segments]) for t in self.tracks]

        print(f"# min_segments {min_segments}")

    def make_window_batches(self, batch_size:int=30): # group for each trackset
        # track_size = batch_size
        # T_0 [  ][  ][  ]
        # T_1 [t0][t1][t2]
        # T_2 [  ][  ][  ]
        #        +
        # T_3 [  ][  ][  ]
        # T_4 [t0][t1][t2]
        # T_4 [  ][  ][  ]
        batch_groups=[]
        for track_group in parse_to_subarrays(self.tracks, batch_size):
            batches=[]
            for s in range(len(track_group[0])): # for t
                batches.append(Batch([track[s] for track in track_group] , tag='window'))
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

    def calulcate_segments_gains(self, gain_model):
        for track in self.tracks:
            gain=gain_model.predict(track)
            for segment in track:
                segment.wet_gain = gain

    def calulcate_segments_noise_profile(self):
        for track in self.tracks:
            noise_profile = estimate_noise_profile(track.get_wet(), self.sample_rate) 
            for segment in track:
                segment.noise_profile = noise_profile
