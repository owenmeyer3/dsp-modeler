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
import os, torch, json
import torch.optim as optim
from scipy.io import wavfile
import numpy as np
from black_box.model.model import ConditionedLSTM, make_conditioned_input, normalize_params, combined_loss
from common.alignment import estimate_shift, apply_shift


def load_wav(path):
    sr, raw = wavfile.read(path)
    data = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        data = data / float(2 ** (raw.dtype.itemsize * 8 - 1))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


class PedalDataset(torch.utils.data.Dataset):
    """Loads the single shared dry reference (DRY_FILENAME) once, then
    pairs it against every OTHER wav file in data_dir -- each of those
    is treated as a wet take for some parameter combination. Since the
    dry signal is identical for every take, it's only loaded/trimmed a
    single time and reused across all wet files."""
 
    def __init__(
        self, 
        train_manifest,
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
        sr, dry_full = load_wav(dry_filename)

        # preprocess
        n_trim = int(silent_lead_in_seconds * sr)
        dry_full = dry_full[n_trim:]
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
            wet_sr, wet_full = load_wav(wr['wet_file'])
            assert wet_sr == sr, f"{wr['wet_file']} has a different sample rate than {dry_filename}"

            # preprocess
            wet_full = wet_full[n_trim:]

            # parse wet filenames to params
            n_param_val_dict = normalize_params(wr['params'], param_configs)

            # Each wet take is its own separate reamp (play + record) pass,
            # so its dry/wet alignment can differ slightly from other takes
            # (USB buffer/driver start-up quantization), even though it
            # stays fixed within a single pass -- see
            # tests/check_alignment.py for the diagnostic history. Estimate
            # and correct that per-file shift before chunking, rather than
            # assuming dry/wet are already sample-aligned.
            shift = estimate_shift(dry_full, wet_full, sr)
            print(f"  {os.path.basename(wr['wet_file'])}: shift = {shift} samples ({shift/sr*1000:.3f} ms)")
            dry_aligned, wet_aligned = apply_shift(dry_full, wet_full, shift)

            n = len(dry_aligned)
            n_chunks = n // chunk_len
            for i in range(n_chunks):
                s = i * chunk_len
                self.chunks.append((
                    dry_aligned[s:s + chunk_len],
                    wet_aligned[s:s + chunk_len],
                    n_param_val_dict
                ))
 
        self.sr = sr
        print(f"Loaded 1 dry reference ({dry_filename}) x {len(wr['wet_file'])} wet takes -> {len(self.chunks)} chunks of {chunk_seconds*1000:.0f}ms each")


    def __len__(self):
        return len(self.chunks)
 
    def __getitem__(self, idx):
        dry, wet, n_param_val_dict = self.chunks[idx]

        param_tensors = []
        for pn in self.param_names:
            val = n_param_val_dict[pn]
            dtype = self.param_configs[pn]['dtype']
            param_tensors.append(torch.tensor(val, dtype=dtype))
        
        tensors = tuple(
            [torch.from_numpy(dry.copy()), torch.from_numpy(wet.copy())] + \
            param_tensors \
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
    train_manifest = '/Users/owenmeyer/dsp-modeler/data/train/manifest.json',
    dry_filename = '/Users/owenmeyer/dsp-modeler/data/input/input.wav',
    device = 'mps' if torch.backends.mps.is_available() else 'cpu',
    model_output_dir=''

):
    print(f"device {device}")
    os.makedirs(model_output_dir, exist_ok=True)
    dataset = PedalDataset(
         train_manifest=train_manifest, 
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

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_output_dir}/checkpoint_epoch{epoch+1}.pt')

    torch.save(model.state_dict(), f'{model_output_dir}/model_final.pt')
    return model


# if __name__ == '__main__':
#     train()