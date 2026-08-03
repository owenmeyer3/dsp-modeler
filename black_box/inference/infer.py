"""
infer.py

Run the trained ConditionedLSTM on a dry file with a fixed (d, f, v)
conditioning, for actually listening to the model's output.

Unlike training -- which resets the LSTM's hidden state to zero every
~30ms chunk (no true TBPTT, see train.py) -- this carries hidden state
continuously across the whole file, which is the real deployment
behavior (a plugin processes audio continuously, it doesn't get reset
every 30ms).

A single unrolled call over the whole ~226s file (21.7M timesteps) blew
up to 100GB+ and crashed -- PyTorch's MPS backend doesn't handle nn.LSTM
over sequences that long efficiently (unlike CUDA's fused cuDNN kernel,
it likely falls back to a per-timestep path that doesn't free/reuse
buffers well at this scale). It's also not how this would ever run in
real deployment anyway -- RTNeural processes small streaming blocks, not
one giant call. So this processes the file in bounded-size blocks
instead, explicitly carrying hidden/cell state from one block into the
next -- continuous state evolution, but bounded memory per call.

Unlike training (which trims the silent lead-in before ever seeing the
audio), inference runs over the ENTIRE dry file, lead-in included --
matching real captured wet files (which also include it), so the output
can be directly compared/lined up against a real recording without an
offset. A real-time plugin has no notion of "trim the lead-in" anyway;
it just processes whatever comes in.

Run with: python infer.py
"""


import os
import torch
from scipy.io import wavfile
import numpy as np
from common.utils import load_wav, write_wav

from black_box.model.model import ConditionedLSTM, make_conditioned_input, normalize_params
from black_box.eval.evaluate_prediction import evaluate_prediction

def infer(
        dry_path='/home/ubuntu/dsp-modeler/data/input/input.wav',
        out_path='/home/ubuntu/dsp-modeler/data/predictions/p.wav',
        checkpoint = '/home/ubuntu/dsp-modeler/black_box/model/models/2026-07-31_01-37/model_final.pt',
        block_seconds=0.1,  # matches the order of magnitude training already ran thousands of forward passes at without incident
        param_configs={
            'd': {'min': 1, 'max': 7, 'dtype': torch.float32},
            'f': {'min': 1, 'max': 7, 'dtype': torch.float32},
            'v': {'min': 1, 'max': 7, 'dtype': torch.float32},
        },
        params={"d": 5.0, "f": 5.0, "v": 5.0},
        param_order = ['d', 'f', 'v']
    ):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = ConditionedLSTM(input_size=4, hidden_size=20).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dry, sr = load_wav(dry_path)
    GAIN = 20

    norm_params = normalize_params(params, param_configs)
    param_tensors = [
        torch.tensor([norm_params[p]], dtype=param_configs[p]['dtype'], device=device)
        for p in param_order
    ]

    block_len = int(block_seconds * sr)
    n_blocks = (len(dry) + block_len - 1) // block_len

    print(f"Running inference on {len(dry)} samples ({len(dry)/sr:.1f}s) in "
          f"{n_blocks} blocks of {block_len} samples, "
          f"d={params['d']}, f={params['f']}, v={params['v']} on {device}...")

    preds = []
    hidden = None
    with torch.no_grad():
        for i in range(n_blocks):
            s = i * block_len
            block = dry[s:s + block_len]
            block_t = torch.from_numpy(block).unsqueeze(0).to(device)  # (1, block_len)


            # x = make_conditioned_input(block_t, param_tensors)  # (1, block_len, 4)
            x = make_conditioned_input(block_t*GAIN, param_tensors)
            pred, hidden = model(x, hidden)
            pred = pred / GAIN
            preds.append(pred.squeeze(0).squeeze(-1).cpu().numpy())

            if (i + 1) % 200 == 0:
                print(f"  block {i+1}/{n_blocks}")
                if device == 'mps':
                    torch.mps.empty_cache()

    pred = np.concatenate(preds)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pred_clipped = np.clip(pred, -1.0, 1.0)
    # match the source files' 32-bit (24-in-32) container instead of
    # downconverting to 16-bit
    write_wav(out_path, pred_clipped, sr)
    print(f"Wrote {out_path}")
    print(f"pred stats: min={pred.min():.4f}, max={pred.max():.4f}, std={pred.std():.4f}")

if __name__ == "__main__":
    # infer(
    #     dry_path='/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     out_path='/home/ubuntu/dsp-modeler/data/predictions/p.wav',
    #     checkpoint = '/home/ubuntu/dsp-modeler/black_box/model/models/2026-07-31_03-18/model_final.pt',
    #     block_seconds=0.1,  # matches the order of magnitude training already ran thousands of forward passes at without incident
    #     param_configs={
    #         'd': {'min': 1, 'max': 7, 'dtype': torch.float32},
    #         'f': {'min': 1, 'max': 7, 'dtype': torch.float32},
    #         'v': {'min': 1, 'max': 7, 'dtype': torch.float32},
    #     },
    #     params={"d": 3.0, "f": 3.0, "v": 3.0},
    #     param_order = ['d', 'f', 'v']
    # )

    evaluate_prediction(
        dry_path = '/home/ubuntu/dsp-modeler/data/input/input.wav',
        real_path = '/home/ubuntu/dsp-modeler/data/outputs/1f9fd0a8-1608-41b4-b7e5-2c697b39a4f1.wav',
        pred_path = '/home/ubuntu/dsp-modeler/data/predictions/p.wav',
        output_dir = '/home/ubuntu/dsp-modeler/black_box/eval',
        target_lufs = -23.0
    )