import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

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


def measure_noise_floor(data, sr, duration=8.0):
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


def plot_average_spectrum(signals, sr, out_path, chunk_seconds=8, start_seconds=None):
    """Average frequency content across a chunk, one curve per signal --
    useful for spotting exactly which frequency bands two takes diverge
    in, without needing to eyeball a 2D spectrogram."""
    names = list(signals.keys())
    chunk_len = int(chunk_seconds * sr)
    start = int(start_seconds * sr) if start_seconds is not None else len(signals[names[0]]) // 2

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