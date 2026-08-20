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
        self.gain = None
        self.noise_profile=None

    def __len__(self):
        return len(self.dry_data)
    
    def normalize_params(self, param_configs:dict) -> dict: #{'d': -0.33333333333333337, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
        norm_params={}
        for k, v in self.params.items():
            param_def = param_configs[k]
            if param_def['dtype'] == torch.float32:
                norm_params[k] = 2 * (v - param_def['min']) / (param_def['max'] - param_def['min']) - 1
        return norm_params

    def get_target_tensor(self, device, remove_noise=False)->list[torch.tensor]: # tensor(
                                                                # [-0.0632],
                                                                # [ 0.0642],
                                                                # [ 0.0651],
                                                            # )    
        if isinstance(self.wet_data, np.ndarray):
            return torch.from_numpy(self.wet_data).to(device).unsqueeze(-1)
        else: return None

    def get_features_tensor(self, device, param_names, param_configs)-> torch.tensor: # tensor(
                                                                                            # [-0.0632, -0.3333, -0.3333, ...],
        # get dry tensor                                                                    # [ 0.0642, -0.3333, -0.3333, ...],
        dry_tensor = torch.from_numpy(self.dry_data).to(device)                             # [ 0.0651, -0.3333, -0.3333, ...],
                                                                                            # ...
        # stretch param tensors for each sample in chunk                                )
        norm_params = self.normalize_params(param_configs)
        param_tensors=[torch.tensor(norm_params[pn], dtype=param_configs[pn]['dtype'], device=device) for pn in param_names]
        expanded_params = [param_tensor.repeat(len(dry_tensor)) for param_tensor in param_tensors]

        # combine to columns, transpose to time-space
        features_tensors=[dry_tensor]+expanded_params
        return torch.stack(features_tensors).T

class Batch():
    def __init__(self, chunks, tag='series'): # tag is a debugging term to tell if the batch is a timeseries of chunks or a time windows of chunks across tracks
        self.chunks:list[Chunk]=chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    def append(self, chunk):
        self.chunks.append(chunk)

    def __iter__(self):
        return iter(self.chunks)

    def shuffle(self, seed=42):
        random.seed(seed)
        random.shuffle(self.chunks)


    def get_features_tensor(self, device, param_names, param_configs): # tensor(
                                                                            # [-0.0632, -0.3333, -0.3333, ...],
                                                                            # [0.0642,  -0.3333, -0.3333, ...],
                                                                            # [0.0651,  -0.3333, -0.3333, ...],
                                                                        # )        
                                                                        # 
        chunk_tensors = [c.get_features_tensor(device, param_names, param_configs) for c in self.chunks]
        return torch.stack(chunk_tensors, dim=0) 


    def get_target_tensor(self, device):   # tensor(
                                                # [-0.0632],
                                                # [ 0.0642],
                                                # [ 0.0651],
                                            # )  
        chunk_tensors = [c.get_target_tensor(device) for c in self.chunks]
        return torch.stack(chunk_tensors, dim=0) 


    def get_gains_tensor(self, device, param_names):   # tensor(
                                                    # [0.5, 1, 1, ...],
                                                    # [0.5, 1, 1, ...],
                                                    # [0.5, 1, 1, ...],
                                                # )  
        chunk_tensors=[]
        for chunk in self.chunks:
            gain_row = [chunk.gain]
            for i in param_names:
                gain_row.append(1.0)

            row_tensor = torch.tensor(gain_row, device=device, dtype=torch.float32)

            samples_per_chunk = len(chunk)
            chunk_tensor = torch.stack([row_tensor] * samples_per_chunk)

            chunk_tensors.append(chunk_tensor)

        return torch.stack(chunk_tensors, dim=0)


    def get_tensors(self, device, param_names, param_configs):
        return self.get_features_tensor(device, param_names, param_configs), self.get_target_tensor(device)


class Track():
    def __init__(self, chunks, sr):
        self.chunks:list[Chunk]=chunks
        self.gain = None
        self.noise_profile=None
        self.sample_rate=sr

    def shuffle(self, seed=42):
        random.seed(seed)
        random.shuffle(self.chunks)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    def append(self, chunk):
        self.chunks.append(chunk)

    def __iter__(self):
        return iter(self.chunks)
    
    def get_dry(self):
        return np.concatenate([chunk.dry_data for chunk in self.chunks])
        
    def get_wet(self):
        return np.concatenate([chunk.wet_data for chunk in self.chunks])

    def get_params(self):
        return self.chunks[0].params

    def compute_rms_gain(self):
        rms_d = np.sqrt(np.mean(self.get_dry() ** 2))
        rms_w = np.sqrt(np.mean(self.get_wet() ** 2))
        return rms_w / rms_d

    def compute_noise_profile(self, silent_lead_in_seconds=8):
        silence_region = self.get_wet()[:silent_lead_in_seconds*self.sample_rate]
        self.noise_profile = estimate_noise_profile(silence_region, self.sample_rate)
        for chunk in self.chunks: chunk.noise_profile = self.noise_profile

    def compute_model_gain(self, gain_model):
        self.gain=gain_model.predict(self)
        for chunk in self.chunks: chunk.gain = self.gain

    def denoise_wet_data(self):
        trk_wet_data = spectral_subtract(self.get_wet(), self.noise_profile, self.sample_rate)
        # re-split back into each chunk's original length
        offset = 0
        for chunk in self.chunks:
            n = len(chunk)
            chunk.wet_data = trk_wet_data[offset:offset + n]
            offset += n

    def add_model_gain(self):
        trk_wet_data = self.get_wet() * self.gain
        # re-split back into each chunk's original length
        offset = 0
        for chunk in self.chunks:
            n = len(chunk)
            chunk.wet_data = trk_wet_data[offset:offset + n]
            offset += n

    def inverse_model_gain(self):
        trk_wet_data = self.get_wet() / self.gain
        # re-split back into each chunk's original length
        offset = 0
        for chunk in self.chunks:
            n = len(chunk)
            chunk.wet_data = trk_wet_data[offset:offset + n]
            offset += n

    def add_constant_gain(self, gain):
        trk_wet_data = self.get_wet() * gain
        # re-split back into each chunk's original length
        offset = 0
        for chunk in self.chunks:
            n = len(chunk)
            chunk.wet_data = trk_wet_data[offset:offset + n]
            offset += n

    @staticmethod
    def from_data(sr, chunk_seconds, params, dry_data=None, wet_data=None):

        data_len = max(len(dry_data) if dry_data is not None else 0, len(wet_data) if wet_data is not None else 0)
        if dry_data is None: dry_data = np.zeros(data_len)
        if wet_data is None: wet_data = np.zeros(data_len)

        chunk_len = int(chunk_seconds * sr)

        dry_chunks_data = parse_to_subarrays(dry_data, chunk_len)

        wet_chunks_data= parse_to_subarrays(wet_data, chunk_len)

        chunks = [Chunk(dry_chunks_data[i].copy(), wet_chunks_data[i].copy(), params) for i in range(len(dry_chunks_data))]

        return Track(chunks, sr)


class DataSet():
    def __init__(self, manifest_file, dry_file, wet_dir, chunk_seconds, param_names, param_configs, silent_lead_in_seconds=8, denoise_wet = True):
        self.tracks:list[Track] = []
        self.sample_rate=None
        self.chunk_seconds = chunk_seconds
        self.param_names = param_names
        self.param_configs = param_configs
        self.silent_lead_in_seconds = silent_lead_in_seconds

        print("loading dataset")

        # Get dry data
        print(f"Load dry: {dry_file}")
        dry_full, sr = load_wav(dry_file)
        self.sample_rate = sr
        n_trim = int(silent_lead_in_seconds * sr)
        dry_trim = dry_full[n_trim:]
        chunk_len = int(chunk_seconds * sr)
        print(f"chunk_len {chunk_len}")
        print(f"sr {sr}")

        # Get wet data
        with open(manifest_file, 'r') as f:
            man_records = [json.loads(line) for line in f if line.strip()]
        for i, man_record in enumerate(man_records):
            wet_file = wet_dir + '/' + man_record['id'] + '.wav'
            print(f"Load wet ({i}/{len(man_records)}): {wet_file}")
            wet_full, wet_sr = load_wav(wet_file)
            assert wet_sr == sr, f"{wet_file} has a different sample rate than {dry_file}"
            wet_trim = wet_full[n_trim:]

            # Get params
            params=man_record['params']


            # dry_aligned, wet_aligned = wet_trim, dry_trim

            # Align dry and wet
            delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, cluster_window_seconds=0.01, verbose=False)
            print(f"delay_samples {delay_samples}")
            dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)

            dry_chunks_data = parse_to_subarrays(dry_aligned, chunk_len)
            wet_chunks_data= parse_to_subarrays(wet_aligned, chunk_len)
            chunks = [Chunk(dry_chunks_data[i].copy(), wet_chunks_data[i].copy(), params) for i in range(len(dry_chunks_data))]
            self.tracks.append(Track(chunks, sr))

            del dry_aligned, wet_aligned, wet_full   # explicitly drop the full-track arrays now that chunking is done

        print(f"# Chunks / Track {len(chunks)}")
        print(f"# Tracks {len(self.tracks)}")

        # resize tracks to equal length
        min_chunks = min([len(track) for track in self.tracks]) - 1 # -1 cuts off any partial chunks

        self.tracks = [Track(t[:min_chunks], sr) for t in self.tracks]

        if denoise_wet:
            self.denoise_wet_data()

        print(f"# min_chunks {min_chunks}")

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

        # Flatten chunks
        chunks = []
        for track in self.tracks:
            for chunk in track:
                chunks.append(chunk)

        # randomize chunks
        random.seed(seed)
        random.shuffle(chunks)

        # batch chunks
        batches = []
        for batch_chunks in parse_to_subarrays(chunks, batch_size):
            batches.append(Batch(batch_chunks, tag='random'))
        return batches
    
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]

    def append(self, track):
        self.tracks.append(track)

    def __iter__(self):
        return iter(self.tracks)

    def compute_model_gain(self, gain_model):
        for track in self.tracks:
            track.compute_model_gain(gain_model)

    def calculate_noise_profiles(self):
        for track in self.tracks:
            track.compute_noise_profile()

    def compute_rms_gain(self):
        for track in self.tracks:
            track.compute_rms_gain()

    def denoise_wet_data(self):
        for track in self.tracks:
            track.denoise_wet_data()

    def add_model_gain(self):
        for track in self.tracks:
            track.add_model_gain()