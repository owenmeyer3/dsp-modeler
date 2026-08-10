"""
model.py

A conditioned LSTM model for black-box pedal/amp modeling, structured to
stay compatible with RTNeural (LSTM + Dense/Linear layers only -- no
custom ops that RTNeural can't load).

Conditioning approach: your 3 knob values (volume, distortion, filter)
are concatenated onto the audio sample at every timestep before the
LSTM sees it. This is the same mechanism used by GuitarML's proven
conditioned models for real-time RTNeural deployment -- simpler and
better-tested for this deployment path than FiLM-style conditioning.

Input to the model at each timestep: [audio_sample, volume, distortion, filter]
  -> shape (batch, seq_len, 4)
Output: predicted audio sample
  -> shape (batch, seq_len, 1)
"""

import torch
import torch.nn as nn

"""
LSTM 
Act Fns
- Sigmoid activation fn (y[0,1] x[-inf,inf], (0, 0.5))
- Hyp. Tangent activation fn (y[-1,1] x[-inf,inf], (0, 0))

States
- Cell State c_n (Long-term momory)
- Hidden State h_n (Short-term memory)

seq_len = samples per chunk
input_size = 1 audio sample + num conditioning params

Chunk slice of input data [input_sample] + [params]
[
    [0.0, -0.3, -0.3, -0.3],
    [0.5, -0.3, -0.3, -0.3],
    [0.1, -0.3, -0.3, -0.3],
    [0.5, -0.3, -0.3, -0.3],
]

Batch
[
    [
        [0.0, -0.3, -0.3, -0.3],
        [0.5, -0.3, -0.3, -0.3],
        [0.1, -0.3, -0.3, -0.3],
        [0.5, -0.3, -0.3, -0.3]
    ],
    [
        [0.0, -0.1, -0.1, -0.1],
        [-0.5, -0.1, -0.1, -0.1],
        [1, -0.1, -0.1, -0.1],
        [-0.5, -0.1, -0.1, -0.1]
    ],
    ...
]
"""


class ConditionedLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=20, num_layers=1):
        """
        input_size: 1 audio sample + N conditioning parameters (3 for RAT: volume, distortion, filter)
        hidden_size: LSTM hidden units. 20 is GuitarML's well-tested default for real-time performance on modest CPUs 
            -- a reasonable starting point
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x, states=None):
        """
        seq_len = samples per chunk
        input_size = 1 audio sample + num conditioning params
        x: (batch, seq_len, input_size) -- audio sample + conditioning params at every timestep
        states: optional (h_0, c_0) for continuing across truncated-backprop-through-time chunks
        returns: (output, (h_0, c_0)) where output is (batch, seq_len, 1)
        """
        out, states = self.lstm(x, states)
        out = self.dense(out)
        return out, states

def make_conditioned_input(dry, params):
    """
    Build the (batch, seq_len, 4) input tensor from a raw audio chunk and
    scalar-per-example conditioning values.

    dry:      (batch, seq_len) raw dry audio samples
    volume, distortion, filt: (batch,) parameter values, already
                normalized to roughly [-1, 1] or [0, 1] -- pick one
                convention and use it consistently everywhere (see
                normalize_param() below).
    """
    batch, seq_len = dry.shape
    dry = dry.unsqueeze(-1)  # (batch, seq_len, 1)

    # (batch,) -> (batch, seq_len, 1), same value repeated across time
    expanded_params = [p.view(batch, 1, 1).expand(batch, seq_len, 1) for p in params]

    return torch.cat([dry] + expanded_params, dim=-1)


def normalize_params(params, param_configs):
    norm_params={}
    for k, v in params.items():
        param_def = param_configs[k]
        if param_def['dtype'] == torch.float32:
            norm_params[k] = 2 * (v - param_def['min']) / (param_def['max'] - param_def['min']) - 1
        else:
            pass
    return norm_params


# ---------------------------------------------------------------------------
# ESR + DC loss -- standard combination for this task (used in the
# GuitarML/Wright lineage of work, and the anti-aliasing fine-tuning
# paper referenced alongside it)
# ---------------------------------------------------------------------------
# def esr_loss(pred, target, eps=1e-8):
#     """Error-to-Signal Ratio: normalizes error by the target's own energy,
#     so the loss is scale-invariant (a quiet passage and a loud passage
#     are weighted fairly, rather than the loss being dominated by
#     whichever happens to be louder)."""
#     return torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + eps)

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

