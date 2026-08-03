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
import os, torch, json, uuid, datetime
import torch.optim as optim
from scipy.io import wavfile
import numpy as np
from black_box.model.model import ConditionedLSTM, make_conditioned_input, normalize_params, combined_loss
# from common.alignment import estimate_shift, apply_shift
from common.delay_ops import measure_delay, apply_shift
from common.utils import load_wav
import eval.spectral_compare as sc

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
            wet_path = wet_dir + '/' + wr['id'] + '.wav'
            wet_full, wet_sr = load_wav(wet_path)
            assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"

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
                    n_param_val_dict
                ))


    def __len__(self):
        return len(self.chunks)
 
    def __getitem__(self, idx):
        dry, wet, n_param_val_dict = self.chunks[idx]

        param_tensors = []
        for pn in self.param_names:
            val = n_param_val_dict[pn]
            dtype = self.param_configs[pn]['dtype']
            param_tensors.append(torch.tensor(val, dtype=dtype))
        
        # tensors = tuple(
        #     [torch.from_numpy(dry.copy()), torch.from_numpy(wet.copy())] + \
        #     param_tensors \
        # )
        tensors = tuple(
            [torch.from_numpy(dry.copy()), torch.from_numpy(wet.copy())]
        )
        return tensors


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
    model_output_dir=''

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

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
        print(f"Epoch {epoch+1}/{epochs}: ESR+DC loss = {avg_loss:.5f}")

        if (epoch + 1) % 5 == 0:
            print(f"  pred mean/std: {pred.mean().item():.5f}/{pred.std().item():.5f} target mean/std: {target.mean().item():.5f}/{target.std().item():.5f}")

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
    trim_noise=True
):
    """
    Diagnostic variant of train(): processes ONE file's chunks in
    sequential order (no shuffling, no batching across files) and carries
    the LSTM's hidden state from one chunk to the next, detaching between
    chunks to truncate backprop -- real TBPTT, matching what this file's
    own top docstring claims happens. hidden is reset to None only once,
    at the start of each epoch's pass over the file.

    Assumes train_manifest's first line is the file to test against.
    """
    print(f"device {device}")
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)

    # Get dry data
    dry_full, sr = load_wav(dry_filename)
    n_trim = int(silent_lead_in_seconds * sr)
    dry_trim = dry_full[n_trim:]
    chunk_len = int(chunk_seconds * sr)

    # Get wet data
    with open(train_manifest, 'r') as f:
        record = json.loads(f.readline())
    wet_path = wet_dir + '/' + record['id'] + '.wav'
    wet_full, wet_sr = load_wav(wet_path)
    if trim_noise:
        noise_profile = sc.estimate_noise_profile(wet_full[:n_trim], wet_sr) # pre-silence
        wet_full = sc.spectral_subtract(wet_full, noise_profile, sr)
    assert wet_sr == sr, f"{wet_path} has a different sample rate than {dry_filename}"
    wet_trim = wet_full[n_trim:]



    # shift dry and wet (1 wet only) - cut samples to return overlap only
    delay_samples, sr = measure_delay(wet_trim, dry_trim, sr, verbose=False)
    dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, delay_samples)
    n = len(dry_aligned)
    n_chunks = n // chunk_len # number of full chunks in data

    print(f"Sequential single-file run: {n_chunks} chunks of {chunk_seconds*1000:.0f}ms each, hidden state carried across all of them")

    n_param_val_dict = normalize_params(record['params'], param_configs)
    param_tensors = [
        torch.tensor([n_param_val_dict[pn]], dtype=param_configs[pn]['dtype'], device=device)
        for pn in param_names
    ]
    input_size = len(param_names) + 1
    model = ConditionedLSTM(input_size=input_size, hidden_size=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    GAIN = 1.0
    for epoch in range(epochs):
        model.train() # puts model into training mode / keep inside the loop incase eval .eval() is used later in loop
        states = None  # hidden state and cell state (h_n, c_n)
        epoch_loss = 0.0
        pred_sum = pred_sq_sum = target_sum = target_sq_sum = n_vals = 0.0

        accum_preds, accum_targets = [], []
        n_backward_steps = 0

        for i in range(n_chunks):
            s = i * chunk_len
            dry_chunk = torch.from_numpy(dry_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)
            wet_chunk = torch.from_numpy(wet_aligned[s:s + chunk_len].copy()).unsqueeze(0).to(device)

            # x = make_conditioned_input(dry_chunk, param_tensors)
            x = make_conditioned_input(dry_chunk*GAIN, param_tensors)
            target = wet_chunk.unsqueeze(-1)

            # Model call
            # h_n & c_n shape = (num_layers, batch, hidden_size) -> (1, 1, 20)
            # First chunk gets tensors for each hidden node and the prediction (to calculate loss)
            # (output, (h_n, c_n)) = model(x, None)
            pred, states = model(x, states)
            pred = pred / GAIN

            # For truncated BPTT
            # take each of h_n and c_n and creates a new tensor with the identical numeric value, but with the 
            # graph connection back through this chunk's LSTM computation cut off
            states = tuple(s.detach() for s in states)  # keep only the previous gradient step in memory

            # in first iteration only
            # drop warmup samples if first chunk otherwise get all samples
            pred_for_loss = pred[:, warmup_samples:, :] if i == 0 else pred
            target_for_loss = target[:, warmup_samples:, :] if i == 0 else target

            # LOSS FUNCTION SEES ~1s OF STEPS
            accum_preds.append(pred_for_loss)
            accum_targets.append(target_for_loss)

            # torch.no_grad() is a context manager — everything inside the with block runs without PyTorch building a computation graph 
            # for it, regardless of whether the tensors involved have requires_grad=True
            # accumulating running totals so you can print pred_mean/pred_std/target_mean/target_std at the end of the epoch
            with torch.no_grad():
                pred_sum += pred_for_loss.sum().item()
                pred_sq_sum += pred_for_loss.pow(2).sum().item()
                target_sum += target_for_loss.sum().item()
                target_sq_sum += target_for_loss.pow(2).sum().item()
                n_vals += pred_for_loss.numel()

            # Only apply gradients after accumulating multiple chunks in order to avoid less certain gradients at low volumes
            # apply new gradients to weights if reached accum_chunks or last chunk
            accum_chunks = int(1 / chunk_seconds) # ~1s at chunk_seconds=0.03
            remaining_after_this = n_chunks - (i + 1) # dont submit if there is not a full chunk after this
            if (len(accum_preds) == accum_chunks and remaining_after_this >= accum_chunks) or i == n_chunks - 1:
                loss = combined_loss(torch.cat(accum_preds, dim=1), torch.cat(accum_targets, dim=1))
                optimizer.zero_grad() # clear old gradients
                loss.backward() # compute fresh gradients for this accumulated window only
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
                optimizer.step() # apply them to the weights
                epoch_loss += loss.item()
                n_backward_steps += 1
                accum_preds, accum_targets = [], []
        avg_loss = epoch_loss / n_backward_steps

        scheduler.step(avg_loss) # gradient step

        pred_mean = pred_sum / n_vals
        pred_std = (pred_sq_sum / n_vals - pred_mean ** 2) ** 0.5
        target_mean = target_sum / n_vals
        target_std = (target_sq_sum / n_vals - target_mean ** 2) ** 0.5

        print(f"Epoch {epoch+1}/{epochs}: ESR+DC loss = {avg_loss:.5f}  lr = {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  pred mean/std: {pred_mean:.5f}/{pred_std:.5f}  target mean/std: {target_mean:.5f}/{target_std:.5f}")
    torch.save(model.state_dict(), f'{model_v_output_dir}/model_final.pt')
    return model


if __name__ == '__main__':

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

    model = train_stateful_single_file(
        wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
        learning_rate=5e-4,
        epochs=40,
        warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
        silent_lead_in_seconds=8,
        chunk_seconds=0.03,
        param_names=["d", "f", "v"],
        param_configs={
            'd':{'min':1, 'max':7, 'dtype':torch.float32},
            'f':{'min':1, 'max':7, 'dtype':torch.float32},
            'v':{'min':1, 'max':7, 'dtype':torch.float32},
        },
        train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train/manifest.jsonl',
        dry_filename='/home/ubuntu/dsp-modeler/data/input/input.wav',
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models'
    )