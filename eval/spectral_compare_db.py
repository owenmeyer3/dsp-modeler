import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
test_file_dir = '/Users/owenmeyer/dsp-modeler/test_files'
paths = {
    'dry': f"{test_file_dir}/input.wav",
    'wet_v3': f"{test_file_dir}/v_3_output.wav",
    'wet_v4': f"{test_file_dir}/v_4_output.wav",
    'wet_v7': f"{test_file_dir}/v_7_output.wav"
}
TARGET_DBFS = -23.0          # common level to normalize each track db to (comes from a tv standard, doesnt matter the value technically)
NULL_TEST_PAIR = ('wet_v4', 'wet_v7')
OUTPUT_DIR = '/Users/owenmeyer/dsp-modeler/test_results'

SILENT_LEADIN_SECONDS = 8.0   # how much guaranteed silence starts every file
ADAPTIVE_GATE_MARGIN_DB = 10  # gate = this file's own noise floor + margin
TRIM_LEADIN_BEFORE_ANALYSIS = True

# ---------------------------------------------------------------------------
# Plain dBFS measurement -- this is the whole "engine" of the script
# ---------------------------------------------------------------------------
def power_to_db(power):
    """Convert signal power (mean of squared samples) to decibels.
    10*log10(power) is standard for power quantities. (If you were
    converting amplitude/RMS directly instead of power, you'd use
    20*log10(amplitude) -- same thing, since power = amplitude^2 and
    10*log10(x^2) = 20*log10(x).)"""
    return 10 * np.log10(power + 1e-15)  # + tiny number avoids log(0)


def block_powers(data, sr, block_ms=400, hop_ms=100):
    """Split the signal into overlapping time windows and compute the
    average power in each window. This is what lets us track loudness
    changing *over time* instead of collapsing the whole file into one
    number immediately -- e.g. so a loud verse and a quiet outro don't
    just get blended into a single meaningless average.

    block_ms: window length in milliseconds (400ms is a reasonable
              default -- long enough to average out individual sample
              noise, short enough to track real changes in level)
    hop_ms:   how far the window moves each step (100ms = 75% overlap
              between consecutive windows, which smooths the result)
    """
    block_size = int(block_ms / 1000 * sr)
    hop = int(hop_ms / 1000 * sr)

    powers = []
    for start in range(0, len(data) - block_size, hop):
        window = data[start:start + block_size]
        powers.append(np.mean(window ** 2))   # mean squared amplitude = power

    return np.array(powers)


def measure_noise_floor(data, sr, duration=SILENT_LEADIN_SECONDS):
    """Look at just the first `duration` seconds (your known-silent
    lead-in) and report its average level in dB. No gating needed here
    since we already know for certain this region has no played signal."""
    n_samples = int(duration * sr)
    lead_in = data[:n_samples]
    powers = block_powers(lead_in, sr)
    return power_to_db(np.mean(powers))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_spectrograms(signals, sr, out_path, chunk_seconds=8, fmax=12000):
    """Spectrograms don't need any of the dB/gating machinery above --
    this is a direct time-vs-frequency view of the raw waveform, made by
    taking an FFT of many short, overlapping windows and stacking them
    side by side. Nothing here is 'adjusted' beyond whatever level-
    matching gain we already applied to the whole signal beforehand."""
    names = list(signals.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(signals[names[0]]) // 2

    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    im = None
    for ax, name in zip(axes, names):
        chunk = signals[name][start:start + chunk_len]
        f_, t_, Sxx = signal.spectrogram(chunk, fs=sr, nperseg=2048, noverlap=1536)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        im = ax.pcolormesh(t_, f_, Sxx_db, shading='auto', cmap='magma', vmin=-100, vmax=-10)
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(name)
        ax.set_ylim(0, fmax)
    axes[-1].set_xlabel('Time (s)')
    fig.colorbar(im, ax=axes, label='dB', location='right', shrink=0.8)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_average_spectrum(signals, sr, out_path, chunk_seconds=8):
    """Average frequency content across a chunk, one curve per signal --
    useful for spotting exactly which frequency bands two takes diverge
    in, without needing to eyeball a 2D spectrogram."""
    names = list(signals.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(signals[names[0]]) // 2

    plt.figure(figsize=(10, 5))
    for name in names:
        chunk = signals[name][start:start + chunk_len]
        freqs, psd = signal.welch(chunk, fs=sr, nperseg=8192)
        plt.semilogx(freqs, 10 * np.log10(psd + 1e-15), label=name)
    plt.xlim(20, sr / 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Average spectrum comparison (level-matched, unweighted dB)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# def load(path):
#     """Read a wav file, return (sample_rate, mono float array in [-1, 1])."""
#     sr, raw = wavfile.read(path)
#     data = raw.astype(np.float64)

#     # Integer PCM (16/24/32-bit) needs scaling down to the [-1, 1] range.
#     # scipy stores samples left-justified in the container dtype, so
#     # dividing by that dtype's own max magnitude gets us back to [-1, 1].
#     if np.issubdtype(raw.dtype, np.integer):
#         max_val = float(2 ** (raw.dtype.itemsize * 8 - 1))
#         data = data / max_val

#     if data.ndim > 1:            # stereo -> mono
#         data = data.mean(axis=1)

#     return sr, data





# def integrated_dbfs(data, sr, gate_db):
#     """The overall level of the *played* content, ignoring silence.

#     Why this needs a gate at all: if we just averaged the power of the
#     entire file including silence, a file with lots of silent gaps would
#     come out quieter than one with less silence, even if the actual
#     playing was identical. Gating throws out blocks that are basically
#     just noise/silence, so the result reflects "how loud is this when
#     something is actually happening."

#     gate_db: any block quieter than this is treated as silence
#              and excluded from the average. Pass in something based on
#              THIS file's own measured noise floor (see main() below) so
#              the threshold makes sense regardless of how hot or quiet
#              the file was originally recorded.
#     """
#     powers = block_powers(data, sr)
#     levels = power_to_db(powers)

#     kept = powers[levels > gate_db]
#     if len(kept) == 0:
#         # Nothing passed the gate -- the whole file is quieter than the
#         # gate itself, so just fall back to using everything.
#         kept = powers

#     return power_to_db(np.mean(kept))


# ---------------------------------------------------------------------------
# Null test: how similar are two signals once their levels are matched?
# ---------------------------------------------------------------------------
# def null_test(a, b, sr, chunk_seconds=5, search_seconds=1):
#     """Subtracts one signal from the other and measures what's left over. 
#     If two signals are truly identical, the residual after subtraction is silence. 
#     The louder the residual relative to the original signal, the more the two signals
#     actually differ, independent of how loud each one is."""

#     corr = signal.correlate(b, a, mode='full')
#     lag = np.argmax(np.abs(corr)) - (len(a) - 1)  

#     if lag >= 0:
#         a_aligned = a[:len(a) - lag]
#         b_aligned = b[lag:lag + len(a_aligned)]
#     else:
#         b_aligned = b[:len(b) + lag]
#         a_aligned = a[-lag:-lag + len(b_aligned)]

#     n = min(len(a_aligned), len(b_aligned))
#     a_aligned, b_aligned = a_aligned[:n], b_aligned[:n]

#     residual = a_aligned - b_aligned
#     rms_signal = np.sqrt(np.mean(a_aligned ** 2))
#     rms_residual = np.sqrt(np.mean(residual ** 2))

#     return {
#         'lag_samples': lag,
#         'lag_ms': lag / sr * 1000,
#         'residual_pct': 100 * rms_residual / rms_signal,
#         'residual_db': 20 * np.log10(rms_residual / rms_signal),
#     }
