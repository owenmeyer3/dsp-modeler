import json, torch, random, datetime, copy, os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from black_box.model.model import ConditionedLSTM, combined_loss
from common.utils import load_wav
from eval.spectral_compare import estimate_noise_profile, spectral_subtract
from common.delay_ops import measure_delay, apply_shift
from scipy.stats import skew

def parse_to_subarrays(arr, group_size):
    return [arr[i:i+group_size] for i in range(0, len(arr), group_size)] # slicing past the end of a list just returns whatever's left rather than erroring

def get_symmetry(x):
    x = x.flatten()
    pos = x[x > 0]
    neg = x[x < 0]
    pos_rms = np.sqrt(np.mean(pos**2)) if len(pos) else 0
    neg_rms = np.sqrt(np.mean(neg**2)) if len(neg) else 0
    p99_9 = np.percentile(x, 99.9)
    p0_1 = np.percentile(x, 0.1)
    skewness = skew(x)
    pn_rms = pos_rms/neg_rms
    p999Overp001 = p99_9/abs(p0_1)
    return [skewness, pn_rms, p999Overp001]


##########################################################################################################################################
##########################################################################################################################################

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
    
    def get_dry_tensor(self, device):
        return torch.from_numpy(self.dry_data).unsqueeze(0).to(device)

    def get_wet_tensor(self, device):
        if isinstance(self.wet_data, np.ndarray):
            return torch.from_numpy(self.wet_data).unsqueeze(0).to(device) 
        else: return None

    def get_target_tensor(self, device):
        if isinstance(self.wet_data, np.ndarray):
            return self.get_wet_tensor(device).unsqueeze(-1)
        else: return None

    def get_features_tensor(self, device, param_names, param_configs):
        dry_tensor = self.get_dry_tensor(device)
        # Make dry input (batch, seq_len, 1)
        batch, seq_len = dry_tensor.shape
        dry_view = dry_tensor.unsqueeze(-1)

        # Make param input (batch, seq_len, 1), same value repeated across time
        param_views = [p.view(batch, 1, 1).expand(batch, seq_len, 1) for p in self.get_param_tensors(device, param_names, param_configs)]
        return torch.cat([dry_view] + param_views, dim=-1)


class Segment():
    def __init__(self, chunks, noise_profile=None):
        self.chunks:list[Chunk]=chunks
        self.noise_profile=noise_profile
    
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]
    
    def append(self, chunk):
        self.chunks.append(chunk)

    def __iter__(self):
        return iter(self.chunks)
    
    def get_features_tensors(self, device, param_names, param_configs):
        return torch.cat([chunk.get_features_tensor(device, param_names, param_configs) for chunk in self.chunks], dim=1)

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

    def get_tensors(self, device, param_names, param_configs):
        features_tensors = torch.cat([segment.get_features_tensors(device, param_names, param_configs) for segment in self.segments], dim=0)
        tgt=[segment.get_target_tensors(device) for segment in self.segments]
        target_tensors = torch.cat(tgt, dim=0) if not None in tgt else None
        return [features_tensors, target_tensors]


class Track():
    def __init__(self, segments):
        self.segments:list[Segments]=segments

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

class DataSet():
    def __init__(self, manifest_file, dry_file, wet_dir, chunk_seconds, param_names, param_configs, segment_length=20, silent_lead_in_seconds=8, trim_noise = True):
        self.tracks = []
        self.n_segments = 0
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        track_segments = []

        # Get dry data
        dry_full, sr = load_wav(dry_file)
        n_trim = int(silent_lead_in_seconds * sr)
        dry_trim = dry_full[n_trim:]
        chunk_len = int(chunk_seconds * sr)

        # Get wet data
        with open(manifest_file, 'r') as f:
            man_records = [json.loads(line) for line in f if line.strip()]
        for man_record in man_records:
            wet_file = wet_dir + '/' + man_record['id'] + '.wav'
            print(f"For {wet_file}")
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
            print(f"n_samples {n_samples} w n_chunks {n_chunks}")

            # Make chunks for this track
            chunk_bucket=[]
            segments=[]
            for i in range(n_chunks):
                s = i * chunk_len

                chunk = Chunk(dry_aligned[s:s + chunk_len].copy(), wet_aligned[s:s + chunk_len].copy(), params)

                chunk_bucket.append(chunk)
                if (i+1) % segment_length == 0:
                    segments.append(Segment(chunk_bucket, noise_profile))
                    chunk_bucket = []
            
            self.tracks.append(Track(segments))
        
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


##########################################################################################################################################
##########################################################################################################################################

class PredictionSet():
    def __init__(
        predict_file=None,
        predict_data=None, params=None, sr=None,
        predict_manifest_record=None, predict_dir=None,
        chunk_seconds=0.03,
        silent_lead_in_seconds=8,
        trim_noise=True
    ):
        if predict_file and params:
            predict_data, sr = load_wav(predict_file)
            # use supplied params
        elif predict_data and params and sr:
            pass
        elif predict_manifest_record and predict_dir:
            predict_data, sr = load_wav(predict_dir + '/' + predict_manifest_record['id'] + '.wav')
            params=man_record['params']
        else:
            assert False, 'invalid predict params'


            n_trim = int(silent_lead_in_seconds * sr)


        if trim_noise:
            noise_profile = estimate_noise_profile(predict_data[:int(silent_lead_in_seconds * sr)], sr) # pre-silence
            predict_data = spectral_subtract(predict_data, noise_profile, sr)
        else: noise_profile=None

        # Get noise
        if trim_noise:
            predict_data = spectral_subtract(predict_data, noise_profile, sr)
        chunk_len = int(chunk_seconds * sr)
        n_chunks = len(predict_data) // chunk_len # number of full chunks in data


        for i in range(n_chunks):
            s = i * chunk_len

            chunk = Chunk(dry_aligned[s:s + chunk_len].copy(), wet_aligned[s:s + chunk_len].copy(), params)

            chunk_bucket.append(chunk)
            if (i+1) % segment_length == 0:
                segments.append(Segment(chunk_bucket, noise_profile))
                chunk_bucket = []
        
        self.track = Track(segments)
    
    def make_window_batches(self): # group for each trackset
        track_group=[self.track]

        batches=[]
        for s_i in range(0, self.n_segments):
            batches.append(Batch([track[s_i]], tag='window'))
        return batches


##########################################################################################################################################
##########################################################################################################################################

class ConditionedLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x, states=None):
        out, states = self.lstm(x, states)
        out = self.dense(out)
        return out, states

def esr_loss(pred, target, eps=1e-8, min_energy=1e-4):
    energy = torch.sum(target ** 2)
    return torch.sum((pred - target) ** 2) / (torch.maximum(energy, torch.tensor(min_energy)) + eps)


def dc_loss(pred, target, eps=1e-8):
    """Penalizes any DC offset difference (mean-level mismatch) between
    prediction and target -- ESR alone doesn't strongly constrain this."""
    target_var = torch.var(target) + eps
    return (torch.mean(pred) - torch.mean(target)) ** 2 / target_var

def pos_neg_balance_loss(pred, target, eps=1e-8):
    target_var = torch.var(target) + eps
    pred_pos_rms = torch.sqrt((pred.clamp(min=0) ** 2).mean() + eps)
    pred_neg_rms = torch.sqrt((pred.clamp(max=0) ** 2).mean() + eps)
    target_pos_rms = torch.sqrt((target.clamp(min=0) ** 2).mean() + eps)
    target_neg_rms = torch.sqrt((target.clamp(max=0) ** 2).mean() + eps)
    return ((pred_pos_rms - target_pos_rms) ** 2 + (pred_neg_rms - target_neg_rms) ** 2) / target_var

def combined_loss(pred, target, esr_weight = 1.0, dc_weight=0.5, pos_neg_weight=0.0):
    esr = esr_loss(pred, target)
    dc = dc_loss(pred, target)
    pos_neg_balance = pos_neg_balance_loss(pred, target)

    return [
        esr + dc_weight * dc + pos_neg_weight * pos_neg_balance,
        esr,
        dc,
        pos_neg_balance
    ]


##########################################################################################################################################
##########################################################################################################################################


def train_manifest(
    wet_dir,
    learning_rate=5e-4,
    epochs=10,
    warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    silent_lead_in_seconds=8,
    chunk_seconds=0.03,
    param_names=["d", "f", "v"],
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    },
    train_manifest='/home/ubuntu/dsp-modeler/data/train/manifest.jsonl',
    validation_manifest='/home/ubuntu/dsp-modeler/data/validation/manifest.jsonl',
    dry_file='/home/ubuntu/dsp-modeler/data/input/input.wav',
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    model_output_dir='',
    trim_noise=True,
    lr_patience = 6,
    lr_factor = 0.5,
    batch_size=30,
    hidden_size=20,
    verbose_time=False,
    verbose_performance = False
):
    start = datetime.datetime.now()
    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)
    
    # Load data
    print(f"Load")
    train_dataset = DataSet(train_manifest, dry_file, wet_dir, chunk_seconds, param_names, param_configs, silent_lead_in_seconds=silent_lead_in_seconds, trim_noise = trim_noise)
    print(f"Loaded 1")
    validation_dataset = DataSet(validation_manifest, dry_file, wet_dir, chunk_seconds, param_names, param_configs, silent_lead_in_seconds=silent_lead_in_seconds, trim_noise = trim_noise)
    print(f"Loaded 2")

    # Model info
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=hidden_size).to(device)
    # model.lstm.flatten_parameters()                                 < =======================================
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    prepped = datetime.datetime.now()
    if verbose_time: print(f"Prep took {prepped - start}")

    # Modeling
    best_loss = float('inf')
    best_state = None
    bad_epochs_count = 0
    print("      |                 LOSS                   |           PREDICTIONS         |         TARGET             ")
    print("EPOCH |    total     esr        dc     pos neg |     mean      std      skew   |    mean       std      skew")
    #     "03/40 |  +27.159   +25.270 / +25.270 / +25.270 |  +0.00018 / +0.00085 / +0.05  |  +0.00000 / +0.00113 / -1.20"
    for epoch in range(epochs):
        e_time = datetime.datetime.now()

        # Training
        model.train()
        for batch in train_dataset.batches_of_random(): # batched segments of no specified track or ts
            # get 
            features_tensors, target_tensors = batch.get_tensors(device, param_names, param_configs)

            pred, _ = model(features_tensors, None)
            pred_for_loss = pred[:, warmup_samples:, :]
            target_for_loss = target_tensors[:, warmup_samples:, :]
            loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, dc_weight=0.5, pos_neg_weight=0.2)
            # Backprop
            optimizer.zero_grad() # clear old gradients
            loss.backward() # compute fresh gradients for this accumulated window only
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
            optimizer.step() # apply them to the weights

        t_time =  datetime.datetime.now()
        if verbose_time: print(f"Train time: {t_time - e_time}")

        model.eval()
        num_tracks = len(validation_dataset.tracks)
        with torch.no_grad():
        # Predictions
            eval_states = None
            eval_preds = [[] for _ in range(num_tracks)]
            eval_targets = [[] for _ in range(num_tracks)]

            # Per Track group
            batch_groups = validation_dataset.make_window_batches(track_size=30)
            for batches in batch_groups:
                for batch in batches:
                    input_batch, target_batch = batch.get_tensors(device,param_names,param_configs)
                    pred_batch, eval_states = model(input_batch, eval_states)
                    # Save pred, tgt in memory structure
                    for i, track in enumerate(batch):
                        eval_preds[i].append(pred_batch[i:i+1])
                        eval_targets[i].append(target_batch[i:i+1])

            p_time =  datetime.datetime.now()
            if verbose_time: print(f"Prediction time: {p_time - t_time}")


        # Validation
            eval_losss = eval_esrs = eval_dcs = eval_pns = pSkewnesss = pPn_rmss = pP999Overp001s = tSkewnesss = tPn_rmss = tP999Overp001s = pMeans = pStds = tMeans = tStds = 0.0
            for p in range(num_tracks):
                eval_pred_p = torch.cat(eval_preds[p], dim=1)
                eval_target_p = torch.cat(eval_targets[p], dim=1)
                eval_loss, eval_esr, eval_dc, eval_pn = combined_loss(eval_pred_p, eval_target_p, dc_weight=0.5, pos_neg_weight=0.2)
                eval_losss += eval_loss.item()
                eval_esrs += eval_esr.item()
                eval_dcs += eval_dc.item()
                eval_pns += eval_pn.item()
                pSkewness, pPn_rms, pP999Overp001 = get_symmetry(eval_pred_p.detach().cpu().numpy())
                pSkewnesss += pSkewness
                pPn_rmss += pPn_rms
                pP999Overp001s += pP999Overp001
                tSkewness, tPn_rms, tP999Overp001 = get_symmetry(eval_target_p.detach().cpu().numpy())
                tSkewnesss += tSkewness
                tPn_rmss += tPn_rms
                tP999Overp001s += tP999Overp001
                pMeans += eval_pred_p.mean().item()
                pStds += eval_pred_p.std().item()
                tMeans += eval_target_p.mean().item()
                tStds += eval_target_p.std().item()
            eval_loss = eval_losss/num_tracks
            eval_esr = eval_esrs/num_tracks
            eval_dc = eval_dcs/num_tracks
            eval_pn = eval_pns/num_tracks
            pMean = pMeans/num_tracks
            pStd = pStds/num_tracks
            pSkewness = pSkewnesss/num_tracks
            tMean = tMeans/num_tracks
            tStd = tStds/num_tracks
            tSkewness = tSkewnesss/num_tracks
            print(f"{(epoch+1):02d}/{epochs:02d} |  {eval_loss:+07.3f}   {eval_esr:+07.3f} / {eval_dc:+07.3f} / {eval_pn:+07.3f} |  {pMean:+0.5f} / {pStd:+0.5f} / {pSkewness:+04.2f}  |  {tMean:+0.5f} / {tStd:+0.5f} / {tSkewness:+04.2f}")

        v_time =  datetime.datetime.now()
        if verbose_time: print(f"Validation time: {v_time - p_time}")


        # Reset model if diverging and lower learning rate
        if eval_loss < best_loss:
            bad_epochs_count = 0
            best_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs_count += 1
            if bad_epochs_count > lr_patience:
                # Restore best model weights
                model.load_state_dict(best_state)
                # change optimizer lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_factor
                bad_epochs_count = 0
                print(f"Plateaued: Restored best weights (L = {best_loss:+9.3f}) with LR {optimizer.param_groups[0]['lr']:.2e}")


    # Get best model and save
    model.load_state_dict(best_state)
    torch.save(best_state, f'{model_v_output_dir}/model_best.pt')
    return model

def predict_track(checkpoint, track):
    # Model info
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()


def predict_manifest(
    checkpoint='',
    prediction_manifest='/home/ubuntu/dsp-modeler/data/prediction/manifest.jsonl',
    predict_dir='/home/ubuntu/dsp-modeler/data/outputs',
    param_names=["d", "f", "v"],
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    },
    silent_lead_in_seconds=8,
    chunk_seconds=0.03,
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    trim_noise=True,
    track_size=30,
    hidden_size=20
):
    prediction_dataset = PredictionSet(
        predict_file=None,
        predict_data=None, params=None, sr=None,
        predict_manifest_record=None, predict_dir=None,
        chunk_seconds=0.03,
        silent_lead_in_seconds=8,
        trim_noise=True
    )

    # Model info
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    num_tracks = len(validation_dataset.tracks)
    with torch.no_grad():
    # Predictions
        eval_states = None
        eval_preds = [[] for _ in range(num_tracks)]

        # Per Track group
        batches = prediction_dataset.make_window_batches(track_size=track_size)[0]
        for batch in batches:
            input_batch, _ = batch.get_tensors(device,param_names,param_configs)
            pred_batch, eval_states = model(input_batch, eval_states)
            # Save pred, tgt in memory structure
            for i, track in enumerate(batch):
                eval_preds[i].append(pred_batch[i:i+1])
    
    eval_pred_p = torch.cat(eval_preds[p], dim=1)

    return eval_pred_p

##########################################################################################################################################
##########################################################################################################################################

if __name__ == '__main__':
    train_manifest(
        wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
        learning_rate=5e-4,
        epochs=20,
        warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
        silent_lead_in_seconds=8,
        chunk_seconds=0.03,
        param_names=["d", "f", "v"],
        param_configs={
            'd':{'min':1, 'max':7, 'dtype':torch.float32},
            'f':{'min':1, 'max':7, 'dtype':torch.float32},
            'v':{'min':1, 'max':7, 'dtype':torch.float32},
        },
        train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl',
        validation_manifest='/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl',
        dry_file='/home/ubuntu/dsp-modeler/data/input/input.wav',
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models',
        trim_noise=True,
        lr_patience = 6,
        lr_factor = 0.5,
        verbose_time=False,
        verbose_performance = False,
        hidden_size=60
    )