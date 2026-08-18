"""
train.py

Training skeleton for the ConditionedLSTM model, using truncated
backpropagation through time (TBPTT). Your files are multi-minute audio
at 96kHz -- way too long to backprop through in one shot (both memory
and gradient-stability reasons), so we chop each file into short chunks,
carry the LSTM's hidden state across chunks within the same file, but
only backprop within each chunk.

Expects a file naming convention like: v3_d5_f2_dry.wav / v3_d5_f2_wet.wav
-- adjust `parse_params_from_filename` to match whatever convention you
actually used when capturing your dataset.
"""
import os, torch, json, uuid, datetime, copy, random
import torch.optim as optim
from scipy.io import wavfile
import numpy as np
from black_box.model.model import ConditionedLSTM, make_conditioned_input, normalize_params, combined_loss
# from common.alignment import estimate_shift, apply_shift
from common.delay_ops import measure_delay, apply_shift
from common.utils import load_wav
import eval.spectral_compare as sc
from scipy.stats import skew
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

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

class PedalDataset(torch.utils.data.Dataset):
    """Loads the single shared dry reference (DRY_FILENAME) once, then
    pairs it against every OTHER wav file in data_dir -- each of those
    is treated as a wet take for some parameter combination. Since the
    dry signal is identical for every take, it's only loaded/trimmed a
    single time and reused across all wet files."""
 
    def __init__(
        self, 
        train_manifest,
        wet_dir,
        param_names,
        dry_filename, 
        param_configs,
        chunk_seconds=0.1,
        silent_lead_in_seconds=8 
    ):
        self.param_names=param_names
        self.param_configs=param_configs
        self.chunks = []  # list of (dry_chunk, wet_chunk, v, d, f)

        # load dry wav
        dry_full, sr = load_wav(dry_filename)

        # preprocess
        n_trim = int(silent_lead_in_seconds * sr)
        dry_trim = dry_full[n_trim:]
        chunk_len = int(chunk_seconds * sr)

        # get training records
        wet_records = []
        with open(train_manifest, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                wet_records.append(record)

        for wr in wet_records:
            wet_params = wr['params']
            wet_path = wet_dir + '/' + wr['id'] + '.wav'
            wet_full, wet_sr = load_wav(wet_path)
            assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"

            noise_profile = sc.estimate_noise_profile(wet_full, wet_sr) # pre-silence
            wet_dn_full = sc.spectral_subtract(wet_full, noise_profile, sr)

            # preprocess
            wet_trim = wet_full[n_trim:]

            delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
            dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
            n = len(dry_aligned)
            n_chunks = n // chunk_len

            for i in range(n_chunks):
                s = i * chunk_len
                self.chunks.append((
                    dry_aligned[s:s + chunk_len],
                    wet_aligned[s:s + chunk_len],
                    wet_params
                ))

    def __len__(self):
        return len(self.chunks)
 
    def __getitem__(self, idx):
        dry, wet, wet_params = self.chunks[idx]

        param_tensors = []
        for pn in self.param_names:
            val = wet_params[pn]
            dtype = self.param_configs[pn]['dtype']
            param_tensors.append(torch.tensor(val, dtype=dtype))
        
        tensors = tuple(
            [torch.from_numpy(dry.copy()), torch.from_numpy(wet.copy())] + \
            param_tensors \
        )





def train(
    batch_size=40,
    learning_rate = 5e-4,
    epochs=100,
    warmup_samples=1000, # let the LSTM "settle" into a chunk before computing loss on it,  so early timesteps with poor hidden state don't dominate the gradient
    silent_lead_in_seconds=8, # matches your capture convention - trim useless noise signal
    chunk_seconds=0.1, # TBPTT chunk length - audio ML commonly uses ~2048-4096 samples; at 96kHz - that's roughly 20-40ms -- start small / tune based on your GPU memory
    param_names=["d", "f", "v"],
    target="dry",
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    },
    train_manifest = '/home/ubuntu/dsp-modeler/data/train/manifest.json',
    dry_filename = '/home/ubuntu/dsp-modeler/data/input/input.wav',
    wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    model_output_dir='',
    verbose_time=False,
    verbose_performance = False
):

    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)
    dataset = PedalDataset(
         train_manifest=train_manifest,
         wet_dir=wet_dir,
         param_names=param_names,
         dry_filename=dry_filename, 
         param_configs=param_configs, 
         chunk_seconds=chunk_seconds,
         silent_lead_in_seconds=silent_lead_in_seconds 
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ConditionedLSTM(input_size=4, hidden_size=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_state = None
    bad_epochs_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for dry, wet, *params in loader:
            dry, wet = dry.to(device), wet.to(device)
            params = [p.to(device) for p in params]
            x = make_conditioned_input(dry, params)
            target = wet.unsqueeze(-1)
            pred, _ = model(x) # hidden=None -> fresh state per chunk; see note below for stateful alternative

            # Skip the warmup region when computing loss so the model
            # isn't penalized before its hidden state has "caught up"
            loss = combined_loss(pred[:, warmup_samples:, :], target[:, warmup_samples:, :])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"{epoch+1}/{epochs}: ESR+DC L = {avg_loss:9.3f}  | P mean/std: {pred.mean().item():.5f}/{pred.std().item():.5f}  | T mean/std: {target.mean().item():.5f}/{target.std().item():.5f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_v_output_dir}/checkpoint_epoch{epoch+1}.pt')

    torch.save(model.state_dict(), f'{model_v_output_dir}/model_final.pt')
    return model

def train_stateful_single_file(
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
    dry_filename='/home/ubuntu/dsp-modeler/data/input/input.wav',
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    model_output_dir='',
    trim_noise=True,
    verbose_time=False,
    verbose_performance = False 
):
    """
    Diagnostic variant of train(): processes ONE file's chunks in
    sequential order (no shuffling, no batching across files) and carries
    the LSTM's hidden state from one chunk to the next, detaching between
    chunks to truncate backprop -- real TBPTT. hidden is reset to None only once,
    at the start of each epoch's pass over the file.

    Assumes train_manifest's first line is the file to test against.
    """
    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)
    best_loss = float('inf')
    best_state = None
    bad_epochs_count = 0

    # Get dry data
    dry_full, sr = load_wav(dry_filename)
    n_trim = int(silent_lead_in_seconds * sr)
    dry_trim = dry_full[n_trim:]
    chunk_len = int(chunk_seconds * sr)

    # Get (opt: noiseless) wet data
    with open(train_manifest, 'r') as f:
        record = json.loads(f.readline())
    wet_path = wet_dir + '/' + record['id'] + '.wav'
    wet_full, wet_sr = load_wav(wet_path)
    if trim_noise:
        noise_profile = sc.estimate_noise_profile(wet_full[:n_trim], wet_sr) # pre-silence
        wet_full = sc.spectral_subtract(wet_full, noise_profile, sr)
    assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"
    wet_trim = wet_full[n_trim:]

    # Process samples, get chunk num
    delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
    dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
    n = len(dry_aligned)
    n_chunks = n // chunk_len # number of full chunks in data
    accum_chunks = int(0.03*20 / chunk_seconds) # ~1s at chunk_seconds=0.03
    print(f"Sequential single-file run: {n_chunks} full chunks of {chunk_seconds*1000:.0f}ms each")

    # Feature info
    n_param_val_dict = normalize_params(record['params'], param_configs)
    param_tensors = [
        torch.tensor([n_param_val_dict[pn]], dtype=param_configs[pn]['dtype'], device=device)
        for pn in param_names
    ]
    input_size = len(param_names) + 1

    # Model info
    model = ConditionedLSTM(input_size=input_size, hidden_size=40).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # lR adjustment settings
    lr_patience = 6
    lr_factor = 0.5

    GAIN = 1.0
    print("      |                 LOSS                   |           PREDICTIONS         |         TARGET             ")
    print("EPOCH |    total     esr        dc     pos neg |     mean      std      skew   |    mean       std      skew")
    #     "03/40 |  +27.159   +25.270 / +25.270 / +25.270 |  +0.00018 / +0.00085 / +0.05  |  +0.00000 / +0.00113 / -1.20"
    for epoch in range(epochs):
        model.train() # puts model into training mode / keep inside the loop incase eval .eval() is used later in loop
        states = None  # hidden state and cell state (h_n, c_n)
        epoch_loss = 0.0
        pred_sum = pred_sq_sum = pred_cube_sum = target_sum = target_sq_sum = target_cube_sum = n_vals = 0.0
        accum_preds, accum_targets = [], []

        for i in range(n_chunks):
            s = i * chunk_len

            # Get chunk features / target
            dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
            wet_chunk = torch.from_numpy(wet_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
            x = make_conditioned_input(dry_chunk*GAIN, param_tensors)
            target = wet_chunk.unsqueeze(-1)

            # Predict (drop warmup samples if first chunk otherwise get all samples)
            # h_n & c_n shape = (num_layers, batch, hidden_size) -> (1, 1, 20)
            # First chunk gets tensors for each hidden node and the prediction (to calculate loss)
            # (output, (h_n, c_n)) = model(x, None)
            pred, states = model(x, states)
            pred = pred / GAIN
            pred_for_loss = pred[:, warmup_samples:, :] if i == 0 else pred
            target_for_loss = target[:, warmup_samples:, :] if i == 0 else target

            # torch.no_grad() is a context manager — everything inside the with block runs without PyTorch building a computation graph 
            # for it, regardless of whether the tensors involved have requires_grad=True
            # accumulating running totals so you can print pred_mean/pred_std/target_mean/target_std at the end of the epoch
            with torch.no_grad():
                pred_sum += pred_for_loss.sum().item()
                pred_sq_sum += pred_for_loss.pow(2).sum().item()
                pred_cube_sum += pred_for_loss.pow(3).sum().item()
                target_sum += target_for_loss.sum().item()
                target_sq_sum += target_for_loss.pow(2).sum().item()
                target_cube_sum += target_for_loss.pow(3).sum().item()
                n_vals += pred_for_loss.numel()

            # LOSS FUNCTION SEES ~1s OF STEPS
            accum_preds.append(pred_for_loss)
            accum_targets.append(target_for_loss)

            # Only apply gradients after accumulating multiple chunks in order to avoid less certain gradients at low volumes
            # apply new gradients to weights if reached accum_chunks or last chunk
            remaining_after_this = n_chunks - (i + 1) # dont submit if there is not a full chunk after this
            if (len(accum_preds) == accum_chunks and remaining_after_this >= accum_chunks) or i == n_chunks - 1:
                states = tuple(s.detach() for s in states)  # keep only the previous gradient step in memory

                loss, _, _, _ = combined_loss(torch.cat(accum_preds, dim=1), torch.cat(accum_targets, dim=1), dc_weight=0.5, pos_neg_weight=0.2)
                optimizer.zero_grad() # clear old gradients
                loss.backward() # compute fresh gradients for this accumulated window only
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
                optimizer.step() # apply them to the weights
                accum_preds, accum_targets = [], []
        
        # Predict validation def validate(n_chunks, chunk_len, device, param_tensors, combined_loss)
        model.eval()
        with torch.no_grad():
            eval_states = None
            eval_preds = []
            for i in range(n_chunks):
                s = i * chunk_len
                dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
                x = make_conditioned_input(dry_chunk, param_tensors)
                pred, eval_states = model(x, eval_states)
                eval_preds.append(pred)
            eval_pred = torch.cat(eval_preds, dim=1)
            eval_target = torch.from_numpy(wet_aligned[:eval_pred.shape[1]].copy()).unsqueeze(0).unsqueeze(-1).to(device)
            eval_loss, eval_esr, eval_dc, eval_pn = combined_loss(eval_pred, eval_target, dc_weight=0.5, pos_neg_weight=0.2)
            eval_loss, eval_esr, eval_dc, eval_pn = eval_loss.item(), eval_esr.item(), eval_dc.item(), eval_pn.item()
            pSkewness, pPn_rms, pP999Overp001 = get_symmetry(eval_pred.detach().cpu().numpy())
            tSkewness, tPn_rms, tP999Overp001 = get_symmetry(eval_target.detach().cpu().numpy())
            pMean = eval_pred.mean().item()
            pStd = eval_pred.std().item()
            tMean = eval_target.mean().item()
            tStd = eval_target.std().item()
            print(f"{(epoch+1):02d}/{epochs:02d} |  {eval_loss:+07.3f}   {eval_esr:+07.3f} / {eval_dc:+07.3f} / {eval_pn:+07.3f} |  {pMean:+0.5f} / {pStd:+0.5f} / {pSkewness:+04.2f}  |  {tMean:+0.5f} / {tStd:+0.5f} / {tSkewness:+04.2f}")
        model.train() # return to train mode

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


def file_to_segments(dry_file, wet_file, params, param_names, param_configs, trim_noise=True, silent_lead_in_seconds=8, chunk_seconds=0.03):
    # Get dry data
    dry_full, sr = load_wav(dry_file)
    n_trim = int(silent_lead_in_seconds * sr)
    dry_trim = dry_full[n_trim:]
    chunk_len = int(chunk_seconds * sr)

    # Get (opt: noiseless) wet data
    wet_full, wet_sr = load_wav(wet_file)
    if trim_noise:
        noise_profile = sc.estimate_noise_profile(wet_full[:n_trim], wet_sr) # pre-silence
        wet_full = sc.spectral_subtract(wet_full, noise_profile, sr)
    assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"
    wet_trim = wet_full[n_trim:]

    # Feature info
    n_param_val_dict = normalize_params(params, param_configs)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    param_tensors = [
        torch.tensor([n_param_val_dict[pn]], dtype=param_configs[pn]['dtype'], device=device)
        for pn in param_names
    ]

    delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
    dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
    n = len(dry_aligned)
    n_chunks = n // chunk_len # number of full chunks in data
    # 22 chunks × 2880 samples/chunk = 63,360 samples, and 23 chunks would be 66,240. 
    # That straddles 65,536 (2^16), which is a well-documented constraint of cuDNN's "persistent" RNN algorithm: 
    # it requires sequence_length × batch_size to stay under roughly that 2^16 ceiling 
    # (a legacy limitation from how the persistent kernels index their per-timestep workspace). 
    # With batch_size=1 here, that reduces to just sequence_length ≲ 65536
    chunks_per_segment = int(0.03 * 20 / chunk_seconds) # ~1s at chunk_seconds=0.03
    n_segments = n_chunks // chunks_per_segment

    # Make chunks to train on
    chunks = []
    for i in range(n_chunks):
        s = i * chunk_len

        # Get chunk features / target
        dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
        wet_chunk = torch.from_numpy(wet_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
        x = make_conditioned_input(dry_chunk, param_tensors)
        target = wet_chunk.unsqueeze(-1)
        chunks.append({
            'dry_chunk':dry_chunk,
            'wet_chunk':wet_chunk,
            'input':x,
            'target':target
        })

    segments = []
    for s in range(n_segments):
        s = s * chunks_per_segment
        segments.append(chunks[s:s + chunks_per_segment])
    
    return segments

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
    verbose_time=False,
    verbose_performance = False
):
    start = datetime.datetime.now()
    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)

    # Get segments from train files
    train_segments=[] # 1d
    with open(train_manifest, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        wet_file = wet_dir + '/' + record['id'] + '.wav'
        params=record['params']
        file_segments = file_to_segments(dry_file, wet_file, params, param_names, param_configs, trim_noise=trim_noise)
        train_segments+=file_segments
    
    # Get ordered segments from validation files
    validation_segmented_tracks = [] # 2d
    with open(validation_manifest, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        wet_file = wet_dir + '/' + record['id'] + '.wav'
        params=record['params']
        file_segments = file_to_segments(dry_file, wet_file, params, param_names, param_configs, trim_noise=trim_noise)
        validation_segmented_tracks.append(file_segments)
    shortest_track_samples = min([len(trk) for trk in validation_segmented_tracks])
    validation_segmented_tracks = [trk[:shortest_track_samples] for trk in validation_segmented_tracks]

    # Shuffle and batch segments
    random.seed(42)
    random.shuffle(train_segments)  # shuffles in place
    batches = [train_segments[i:i+batch_size] for i in range(0, len(train_segments), batch_size)]

    # Model info
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=40).to(device)
    model.lstm.flatten_parameters()
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
        model.train() # puts model into training mode / keep inside the loop incase eval .eval() is used later in loop
        prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if verbose_performance else nullcontext()
        with prof_ctx as prof:

            for batch in batches:  
                # Predict
                x_batch = torch.cat([torch.cat([c['input'] for c in seg], dim=1) for seg in batch], dim=0)
                target_batch = torch.cat([torch.cat([c['target'] for c in seg], dim=1) for seg in batch], dim=0)
                pred, _ = model(x_batch, None)
                pred_for_loss = pred[:, warmup_samples:, :]
                target_for_loss = target_batch[:, warmup_samples:, :]
                loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, dc_weight=0.5, pos_neg_weight=0.2)
                # Backprop
                optimizer.zero_grad() # clear old gradients
                loss.backward() # compute fresh gradients for this accumulated window only
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
                optimizer.step() # apply them to the weights

        if verbose_performance: print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        if verbose_performance: print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        t_time =  datetime.datetime.now()
        if verbose_time: print(f"Train time: {t_time - e_time}")

        # Validation
        model.eval()
        with torch.no_grad():
            eval_states = None
            num_tracks = len(validation_segmented_tracks)
            eval_preds = [[] for _ in range(num_tracks)]
            eval_targets = [[] for _ in range(num_tracks)]

            tracks_by_segment = [list(row) for row in zip(*validation_segmented_tracks)]

            for segments in tracks_by_segment:
                x_block = torch.cat([torch.cat([chunk['input'] for chunk in segment], dim=1) for segment in segments], dim=0)
                target_block = torch.cat([torch.cat([chunk['target'] for chunk in segment], dim=1) for segment in segments], dim=0)
                pred, states = model(x_block, eval_states)
                for p in range(num_tracks):
                    eval_preds[p].append(pred[p:p+1])
                    eval_targets[p].append(target_block[p:p+1])

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
        model.train() # return to train mode

        v_time =  datetime.datetime.now()
        if verbose_time: print(f"Validation time: {v_time - t_time}")

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
    





def train_shuffled_segmented_single_file(
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
    dry_filename='/home/ubuntu/dsp-modeler/data/input/input.wav',
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    model_output_dir='',
    trim_noise=True,
    lr_patience = 6,
    lr_factor = 0.5,
    batch_size=30,
    verbose_time=False,
    verbose_performance = False
):
    start = datetime.datetime.now()
    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)
    best_loss = float('inf')
    best_state = None
    bad_epochs_count = 0

    # Get dry data
    dry_full, sr = load_wav(dry_filename)
    n_trim = int(silent_lead_in_seconds * sr)
    dry_trim = dry_full[n_trim:]
    chunk_len = int(chunk_seconds * sr)

    # Get (opt: noiseless) wet data
    with open(train_manifest, 'r') as f:
        record = json.loads(f.readline())
    wet_path = wet_dir + '/' + record['id'] + '.wav'
    wet_full, wet_sr = load_wav(wet_path)
    if trim_noise:
        noise_profile = sc.estimate_noise_profile(wet_full[:n_trim], wet_sr) # pre-silence
        wet_full = sc.spectral_subtract(wet_full, noise_profile, sr)
    assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"
    wet_trim = wet_full[n_trim:]

    # Feature info
    n_param_val_dict = normalize_params(record['params'], param_configs)
    param_tensors = [
        torch.tensor([n_param_val_dict[pn]], dtype=param_configs[pn]['dtype'], device=device)
        for pn in param_names
    ]

    # Process samples, get chunk num
    delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
    dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
    n = len(dry_aligned)
    n_chunks = n // chunk_len # number of full chunks in data
    # 22 chunks × 2880 samples/chunk = 63,360 samples, and 23 chunks would be 66,240. 
    # That straddles 65,536 (2^16), which is a well-documented constraint of cuDNN's "persistent" RNN algorithm: 
    # it requires sequence_length × batch_size to stay under roughly that 2^16 ceiling 
    # (a legacy limitation from how the persistent kernels index their per-timestep workspace). 
    # With batch_size=1 here, that reduces to just sequence_length ≲ 65536
    chunks_per_segment = int(0.03*20 / chunk_seconds) # ~1s at chunk_seconds=0.03
    n_segments = n_chunks // chunks_per_segment
    
    
    # Make chunks to train on
    chunks = []
    for i in range(n_chunks):
        s = i * chunk_len

        # Get chunk features / target
        dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
        wet_chunk = torch.from_numpy(wet_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
        x = make_conditioned_input(dry_chunk, param_tensors)
        target = wet_chunk.unsqueeze(-1)
        chunks.append({
            'dry_chunk':dry_chunk,
            'wet_chunk':wet_chunk,
            'input':x,
            'target':target
        })

    segments = []
    for s in range(n_segments):
        s = s * chunks_per_segment
        segments.append(chunks[s:s + chunks_per_segment])
    
    batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]

    # Randomly shuffle segments
    random.seed(42)
    random.shuffle(segments)  # shuffles in place
    print(f"Sequential single-file run: {len(chunks)} chunks of {chunk_seconds*1000:.0f}ms each")
    print(f"Sequential single-file run: {len(segments)} segments of {chunks_per_segment} chunks each")
    print(f"Sequential single-file run: {len(batches)} batches of {batch_size} batch_size")


    # Model info
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=40).to(device)
    model.lstm.flatten_parameters()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    prepped = datetime.datetime.now()
    if verbose_time: print(f"Prep took {prepped - start}")

    GAIN = 1.0
    print("      |                 LOSS                   |           PREDICTIONS         |         TARGET             ")
    print("EPOCH |    total     esr        dc     pos neg |     mean      std      skew   |    mean       std      skew")
    #     "03/40 |  +27.159   +25.270 / +25.270 / +25.270 |  +0.00018 / +0.00085 / +0.05  |  +0.00000 / +0.00113 / -1.20" 
    multichunk_predict = False
    multibatch_predict = True
    for epoch in range(epochs):
        model.train() # puts model into training mode / keep inside the loop incase eval .eval() is used later in loop
        e_time = datetime.datetime.now()

        prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if verbose_performance else nullcontext()

        with prof_ctx as prof:

            if multibatch_predict:
                for batch in batches:  
                    x_batch = torch.cat([torch.cat([c['input'] for c in seg], dim=1) for seg in batch], dim=0)
                    target_batch = torch.cat([torch.cat([c['target'] for c in seg], dim=1) for seg in batch], dim=0)
                    pred, _ = model(x_batch, None)
                    pred_for_loss = pred[:, warmup_samples:, :]
                    target_for_loss = target_batch[:, warmup_samples:, :]
                    loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, dc_weight=0.5, pos_neg_weight=0.2)
            
                    optimizer.zero_grad() # clear old gradients
                    loss.backward() # compute fresh gradients for this accumulated window only
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
                    optimizer.step() # apply them to the weights
            
            else:
                for segment in segments:    
                    s_preds, s_targets = [], []

                    if multichunk_predict:
                        x_full = torch.cat([chunk['input'] for chunk in segment], dim=1).contiguous()
                        target_full = torch.cat([chunk['target'] for chunk in segment], dim=1).contiguous()
                        pred, _ = model(x_full, None)
                        pred_for_loss = pred[:, warmup_samples:, :]
                        target_for_loss = target_full[:, warmup_samples:, :]
                        loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, dc_weight=0.5, pos_neg_weight=0.2)
                    else:
                        states = None  # hidden state and cell state (h_n, c_n)
                        first_chunk = True
                        for chunk in segment:
                            x = chunk['input']
                            target = chunk['target']
                            pred, states = model(x, states)
                            pred_for_loss = pred[:, warmup_samples:, :] if first_chunk else pred
                            target_for_loss = target[:, warmup_samples:, :] if first_chunk else target
                            first_chunk = False
                            s_preds.append(pred_for_loss)
                            s_targets.append(target_for_loss)
                        loss, _, _, _ = combined_loss(torch.cat(s_preds, dim=1), torch.cat(s_targets, dim=1), dc_weight=0.5, pos_neg_weight=0.2)

                    optimizer.zero_grad() # clear old gradients
                    loss.backward() # compute fresh gradients for this accumulated window only
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
                    optimizer.step() # apply them to the weights

        if verbose_performance: print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        t_time =  datetime.datetime.now()
        if verbose_time: print(f"Train time: {t_time - e_time}")

        # Predict validation def validate(n_chunks, chunk_len, device, param_tensors, combined_loss)
        model.eval()
        with torch.no_grad():
            eval_states = None
            eval_preds = []
            for i in range(n_chunks):
                s = i * chunk_len
                dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
                x = make_conditioned_input(dry_chunk, param_tensors)
                pred, eval_states = model(x, eval_states)
                eval_preds.append(pred)
            # for block_idx in range(n_blocks):  # same block count for every file
            #     x_block = torch.stack([file_chunks[f][block_idx] for f in range(len(files))], dim=0)  # (n_files, block_len, input_size)
            #     pred, states = model(x_block, states)
            #     for f in range(len(files)):
            #         all_preds[f].append(pred[f:f+1])
            eval_pred = torch.cat(eval_preds, dim=1)
            eval_target = torch.from_numpy(wet_aligned[:eval_pred.shape[1]].copy()).unsqueeze(0).unsqueeze(-1).to(device)
            eval_loss, eval_esr, eval_dc, eval_pn = combined_loss(eval_pred, eval_target, dc_weight=0.5, pos_neg_weight=0.2)
            eval_loss, eval_esr, eval_dc, eval_pn = eval_loss.item(), eval_esr.item(), eval_dc.item(), eval_pn.item()
            pSkewness, pPn_rms, pP999Overp001 = get_symmetry(eval_pred.detach().cpu().numpy())
            tSkewness, tPn_rms, tP999Overp001 = get_symmetry(eval_target.detach().cpu().numpy())
            pMean = eval_pred.mean().item()
            pStd = eval_pred.std().item()
            tMean = eval_target.mean().item()
            tStd = eval_target.std().item()
            print(f"{(epoch+1):02d}/{epochs:02d} |  {eval_loss:+07.3f}   {eval_esr:+07.3f} / {eval_dc:+07.3f} / {eval_pn:+07.3f} |  {pMean:+0.5f} / {pStd:+0.5f} / {pSkewness:+04.2f}  |  {tMean:+0.5f} / {tStd:+0.5f} / {tSkewness:+04.2f}")
        model.train() # return to train mode

        v_time =  datetime.datetime.now()
        if verbose_time: print(f"Validation time: {v_time - t_time}")

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


if __name__ == '__main__':

    # # 1 audio overfit - chunks in ordered accumulated sequences
    # model = train_stateful_single_file(
    #     wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
    #     learning_rate=5e-4,
    #     epochs=20,
    #     warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    #     silent_lead_in_seconds=8,
    #     chunk_seconds=0.03,
    #     param_names=["d", "f", "v"],
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl',
    #     dry_filename='/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models'
    # )

    # 1 audio overfit - chunks in shuffles sequences
    train_shuffled_segmented_single_file(
        wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
        learning_rate=5e-4*30,
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
        dry_filename='/home/ubuntu/dsp-modeler/data/input/input.wav',
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models',
        trim_noise=True,
        lr_patience = 6,
        lr_factor = 0.5
    )

    # Many audio trains - chunked in shuffled sequences
    # train_manifest(
    #     wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
    #     learning_rate=5e-4,
    #     epochs=20,
    #     warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    #     silent_lead_in_seconds=8,
    #     chunk_seconds=0.03,
    #     param_names=["d", "f", "v"],
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl',
    #     validation_manifest='/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl',
    #     dry_file='/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models',
    #     trim_noise=True,
    #     lr_patience = 6,
    #     lr_factor = 0.5,
    #     batch_size=30,
    #     verbose_time=False,
    #     verbose_performance = False
    # )



    # model = train(
    #     batch_size=40,
    #     learning_rate = 5e-3,
    #     epochs=40,
    #     warmup_samples=1000,      # let the LSTM "settle" into a chunk before computing loss on it,  so early timesteps with poor hidden state don't dominate the gradient
    #     silent_lead_in_seconds=8, # matches your capture convention - trim useless noise signal
    #     chunk_seconds=0.03,       # TBPTT chunk length - audio ML commonly uses ~2048-4096 samples; at 96kHz - that's roughly 20-40ms -- start small / tune based on your GPU memory
    #     param_names=["d", "f", "v"],
    #     target="dry",
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     train_manifest =    '/home/ubuntu/dsp-modeler/black_box/data/train/manifest.jsonl',
    #     dry_filename =      '/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models
    # )





    # train_manifest(
    #     wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
    #     learning_rate=5e-4,
    #     epochs=20,
    #     warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    #     silent_lead_in_seconds=8,
    #     chunk_seconds=0.03,
    #     param_names=["d", "f", "v"],
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     train_manifest='/home/ubuntu/dsp-modeler/data/train/manifest.jsonl',
    #     validation_manifest='/home/ubuntu/dsp-modeler/data/validation/manifest.jsonl',
    #     dry_file='/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     model_output_dir='',
    #     trim_noise=True,
    #     lr_patience = 6,
    #     lr_factor = 0.5,
    #     batch_size=30,
    #     verbose_performance = False
    # )

