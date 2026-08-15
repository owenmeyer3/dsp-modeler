import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI needed; saves directly to file
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import stft, istft

# ---------------------------------------------------------------------------
# ITU-R BS.1770 K-weighting + integrated LUFS
# ---------------------------------------------------------------------------
def k_weighting_filter(sr):
    """Return (b, a) biquad coefficients for the two-stage K-weighting filter."""
    # Stage 1: high shelf
    f0, G, Q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b_stage1 = [(Vh + Vb * K / Q + K * K) / a0,
                2 * (K * K - Vh) / a0,
                (Vh - Vb * K / Q + K * K) / a0]
    a_stage1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]

    # Stage 2: RLB high-pass
    f0b, Qb = 38.13547087602444, 0.5003270373238773
    Kb = np.tan(np.pi * f0b / sr)
    a0b = 1 + Kb / Qb + Kb * Kb
    b_stage2 = [1 / a0b, -2 / a0b, 1 / a0b]
    a_stage2 = [1, 2 * (Kb * Kb - 1) / a0b, (1 - Kb / Qb + Kb * Kb) / a0b]

    return (b_stage1, a_stage1), (b_stage2, a_stage2)


def _block_powers(data, sr, weighted=True):
    """Split into 400ms/75%-overlap blocks, return per-block power.
    If weighted=True, applies K-weighting first (perceptual curve).
    If weighted=False, uses the raw signal as-is (plain physical energy)."""
    if weighted:
        (b1, a1), (b2, a2) = k_weighting_filter(sr)
        data = signal.lfilter(b1, a1, data)
        data = signal.lfilter(b2, a2, data)

    block_size = int(0.4 * sr)   # 400ms blocks
    hop = int(block_size * 0.25)  # 75% overlap

    powers = np.array([
        np.mean(data[start:start + block_size] ** 2)
        for start in range(0, len(data) - block_size, hop)
    ])
    return powers[powers > 0]

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

def _to_lufs(p):
    """K-weighted power -> LUFS (includes BS.1770's -0.691 calibration offset)."""
    return -0.691 + 10 * np.log10(p)


def _to_dbfs(p):
    """Plain (unweighted) power -> dBFS. No perceptual offset -- this is
    just the physical energy relative to full scale."""
    return 10 * np.log10(p)

def measure_noise_floor(data, sr, duration=8.0, weighted=True):
    """Loudness of just the known-silent lead-in (no gating needed, since
    we already know this region contains no played signal). Returns LUFS
    if weighted=True, plain dBFS if weighted=False."""
    n = int(duration * sr)
    powers = _block_powers(data[:n], sr, weighted=weighted)
    if len(powers) == 0:
        return -100.0
    mean_power = np.mean(powers)
    return _to_lufs(mean_power) if weighted else _to_dbfs(mean_power)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_spectrograms(normalized, sr, out_path, chunk_seconds=8, fmin=0, fmax=12000, vmin=-100, vmax=-10):
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(normalized[names[0]]) // 2
    start = len(normalized[names[0]]) // 2 + 100000*48

    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    im = None
    for ax, name in zip(axes, names):
        chunk = normalized[name][start:start + chunk_len]
        f_, t_, Sxx = signal.spectrogram(chunk, fs=sr, nperseg=2048, noverlap=1536)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        im = ax.pcolormesh(t_, f_, Sxx_db, shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(name)
        ax.set_ylim(fmin, fmax)
    axes[-1].set_xlabel('Time (s)')
    fig.colorbar(im, ax=axes, label='dB', location='right', shrink=0.8)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_average_spectrum(normalized, sr, out_path, chunk_seconds=8, start_seconds=None):
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(normalized[names[0]]) // 2
    start = int(start_seconds * sr) if start_seconds is not None else len(signals[names[0]]) // 2

    plt.figure(figsize=(10, 5))
    for name in names:
        chunk = normalized[name][start:start + chunk_len]
        freqs, psd = signal.welch(chunk, fs=sr, nperseg=8192)
        plt.semilogx(freqs, 10 * np.log10(psd + 1e-15), label=name)
    plt.xlim(20, sr / 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Average spectrum comparison (LUFS-matched)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def print_band_table(normalized, sr, reference, chunk_seconds=8,
                      bands=(500, 1000, 2000, 3000, 5000, 8000, 10000,
                             15000, 20000, 25000, 30000, 35000, 40000, 45000)):
    """Print power (dB) at specific frequency bands for each signal, welch PSD based."""
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(normalized[reference]) // 2

    specs = {}
    for name in names:
        chunk = normalized[name][start:start + chunk_len]
        freqs, psd = signal.welch(chunk, fs=sr, nperseg=8192)
        specs[name] = (freqs, 10 * np.log10(psd + 1e-15))

    header = ' | '.join(f'{n:>10}' for n in names)
    print(f'{"Freq":>8} | {header}')
    for b in bands:
        idx = np.argmin(np.abs(specs[reference][0] - b))
        row = ' | '.join(f'{specs[n][1][idx]:10.1f}' for n in names)
        print(f'{b:>8} | {row}')


def integrated_level(data, sr, abs_gate, weighted=True):
    """Gated integrated level -- LUFS (weighted=True, per ITU-R BS.1770)
    or plain dBFS (weighted=False), your choice. `abs_gate` must be in
    the same units you're requesting back (LUFS or dBFS respectively)."""
    powers = _block_powers(data, sr, weighted=weighted)
    to_unit = _to_lufs if weighted else _to_dbfs

    gated_abs = powers[to_unit(powers) > abs_gate]        # absolute gate
    if len(gated_abs) == 0:
        return -100.0
    rel_thresh = to_unit(np.mean(gated_abs)) - 10          # relative gate
    gated_rel = gated_abs[to_unit(gated_abs) > rel_thresh]
    if len(gated_rel) == 0:
        gated_rel = gated_abs

    return to_unit(np.mean(gated_rel))


# ---------------------------------------------------------------------------
# Null test: align two signals via cross-correlation, then diff
# ---------------------------------------------------------------------------
def null_test(a, b, sr, chunk_seconds=5, search_seconds=1):
    """Align `b` to `a` (small-lag cross-correlation on a middle chunk),
    then return residual stats. a and b should already be level-matched."""
    chunk_len = int(chunk_seconds * sr)
    search = int(search_seconds * sr)
    start = len(a) // 2

    a_chunk = a[start:start + chunk_len]
    b_search = b[start - search:start + chunk_len + search]

    corr = signal.correlate(b_search, a_chunk, mode='valid')
    lag = np.argmax(np.abs(corr)) - search

    if lag >= 0:
        a_al = a[:len(a) - lag]
        b_al = b[lag:lag + len(a_al)]
    else:
        b_al = b[:len(b) + lag]
        a_al = a[-lag:-lag + len(b_al)]

    n = min(len(a_al), len(b_al))
    a_al, b_al = a_al[:n], b_al[:n]

    residual = a_al - b_al
    rms_a = np.sqrt(np.mean(a_al ** 2))
    rms_res = np.sqrt(np.mean(residual ** 2))

    return {
        'lag_samples': lag,
        'lag_ms': lag / sr * 1000,
        'rms_signal': rms_a,
        'rms_residual': rms_res,
        'residual_pct': 100 * rms_res / rms_a,
        'residual_db': 20 * np.log10(rms_res / rms_a),
        'a_aligned': a_al,
        'b_aligned': b_al,
    }