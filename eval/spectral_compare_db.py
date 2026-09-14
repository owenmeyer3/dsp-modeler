import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import stft, istft

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

def estimate_noise_profile(noise_sample, sr, nperseg=2048):
    """Average magnitude spectrum from a known-silent region -- the
    noise's own frequency-domain fingerprint, across the whole spectrum,
    not just a single band.
    n_freq_bins = nperseg // 2 + 1
    """
    _, _, Zxx = stft(noise_sample, fs=sr, nperseg=nperseg)
    return np.mean(np.abs(Zxx), axis=1, keepdims=True)  # (n_freq_bins, 1)

def spectral_subtract(signal, noise_profile, sr, nperseg=2048, oversubtract=1.8, floor=0.02):
    """Subtract the estimated noise magnitude from signal's STFT, bin by
    bin, then reconstruct. Appropriate for stationary, non-repeating
    (random) analog noise -- unlike time-domain subtraction, which only
    works if the noise repeats exactly sample-for-sample, which real
    analog hiss doesn't."""
    _, _, Zxx = stft(signal, fs=sr, nperseg=nperseg)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    mag_clean = np.maximum(mag - oversubtract * noise_profile, floor * mag)
    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, signal_clean = istft(Zxx_clean, fs=sr, nperseg=nperseg)
    return signal_clean

def spectral_add_noise(signal, noise_profile, sr, nperseg=2048, overadd=1.8, seed=None):
    """Synthesize noise matching noise_profile's magnitude spectrum (random
    phase, appropriate for stationary analog hiss) and add it to signal.
    Not an inverse of spectral_subtract -- that's lossy (the floor-clamp
    discards the original magnitude whenever it wins the max()) -- this
    just reintroduces plausible noise with the right spectral shape."""
    rng = np.random.default_rng(seed)
    _, _, Zxx = stft(signal, fs=sr, nperseg=nperseg)
    n_frames = Zxx.shape[1]

    random_phase = rng.uniform(-np.pi, np.pi, size=(noise_profile.shape[0], n_frames))
    noise_stft = overadd * noise_profile * np.exp(1j * random_phase)
    _, noise_time = istft(noise_stft, fs=sr, nperseg=nperseg)

    n = min(len(signal), len(noise_time))
    return signal[:n] + noise_time[:n]

# def is_track_audible(wet_data, sr, silent_lead_in_seconds=8.0, active_percentile=99, snr_threshold_db=20.0):
#     """Robust audibility check: compares a high percentile of block power
#     (post-lead-in) against the track's own measured noise floor. Percentile-
#     based rather than peak or mean, so a handful of single-sample clicks --
#     which can only ever pollute a few of thousands of 400ms blocks -- can't
#     flip a genuinely silent track to 'audible'."""
#     noise_floor_db = measure_noise_floor(wet_data, sr, duration=silent_lead_in_seconds)

#     n_lead = int(silent_lead_in_seconds * sr)
#     played = wet_data[n_lead:]
#     powers = block_powers(played, sr)
#     if len(powers) == 0:
#         return False, noise_floor_db, -100.0

#     active_level_db = power_to_db(np.percentile(powers, active_percentile))
#     snr_db = active_level_db - noise_floor_db
#     return snr_db >= snr_threshold_db

def is_track_audible(wet_data, sr, silent_lead_in_seconds=8.0, active_percentile=99, snr_threshold_db=20.0, min_active_db=-60.0):
    """Robust audibility check: compares a high percentile of block power
    (post-lead-in) against the track's own measured noise floor. Percentile-
    based rather than peak or mean, so a handful of single-sample clicks --
    which can only ever pollute a few of thousands of 400ms blocks -- can't
    flip a genuinely silent track to 'audible'."""
    noise_floor_db = measure_noise_floor(wet_data, sr, duration=silent_lead_in_seconds)

    n_lead = int(silent_lead_in_seconds * sr)
    played = wet_data[n_lead:]
    powers = block_powers(played, sr)
    if len(powers) == 0:
        return False, noise_floor_db, -100.0

    active_level_db = power_to_db(np.percentile(powers, active_percentile))
    snr_db = active_level_db - noise_floor_db

    # audible = (snr_db >= snr_threshold_db) and (active_level_db >= min_active_db)
    audible = active_level_db >= min_active_db
    # audible = snr_db >= snr_threshold_db

    return audible

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