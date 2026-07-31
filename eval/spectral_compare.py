import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI needed; saves directly to file
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# ---------------------------------------------------------------------------
# CONFIG — edit these for your own files/comparison
# ---------------------------------------------------------------------------
test_file_dir = '/Users/owenmeyer/dsp-modeler/test_files'
paths = {
    'dry': f"{test_file_dir}/input.wav",
    'wet_v4': f"{test_file_dir}/v_4_output.wav",
    'wet_v7': f"{test_file_dir}/v_7_output.wav"
}
TARGET_LUFS = -23.0          # common loudness target for level-matching
NULL_TEST_PAIR = ('wet_v4', 'wet_v7')  # which two keys to null-test against each other
OUTPUT_DIR = '.'             # where to save the .png plots

# Every file is expected to start with this much real silence (hardware
# noise floor, no played signal) -- e.g. 3 measures at 120bpm = 8.0s.
SILENT_LEADIN_SECONDS = 8.0
USE_PERCEPTUAL_WEIGHTING = True  # True: LUFS/K-weighted (ITU-R BS.1770,
                                  #   matches how loud it sounds to a person)
                                  # False: plain dBFS, raw physical energy,
                                  #   no perceptual curve applied
USE_ADAPTIVE_GATE = True     # gate each file relative to its OWN measured
                              # noise floor instead of a fixed -70 LUFS/dBFS
ADAPTIVE_GATE_MARGIN_DB = 10  # gate = noise_floor + this many dB
TRIM_LEADIN_BEFORE_ANALYSIS = True  # drop the silent lead-in before
                                     # spectrograms/null test/etc.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(path):
    """Load a wav file and return (sample_rate, mono float64 array in [-1, 1])."""
    sr, raw = wavfile.read(path)
    data = raw.astype(np.float64)

    # scipy left-justifies < 32-bit PCM into the container dtype, so just
    # scale by the dtype's own max magnitude to get back to [-1, 1].
    if np.issubdtype(raw.dtype, np.integer):
        max_val = float(2 ** (raw.dtype.itemsize * 8 - 1))
        data = data / max_val
    # if it's already float32/float64 PCM, leave as-is

    # collapse to mono if stereo (average channels)
    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data


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


def _to_lufs(p):
    """K-weighted power -> LUFS (includes BS.1770's -0.691 calibration offset)."""
    return -0.691 + 10 * np.log10(p)


def _to_dbfs(p):
    """Plain (unweighted) power -> dBFS. No perceptual offset -- this is
    just the physical energy relative to full scale."""
    return 10 * np.log10(p)


def rms_dbfs(data):
    """Plain, unweighted RMS level in dBFS for a whole array in one shot --
    handy for a quick single-number check, e.g. rms_dbfs(data[:some_slice])."""
    return _to_dbfs(np.mean(data ** 2) + 1e-15)


def measure_noise_floor(data, sr, duration=SILENT_LEADIN_SECONDS, weighted=True):
    """Loudness of just the known-silent lead-in (no gating needed, since
    we already know this region contains no played signal). Returns LUFS
    if weighted=True, plain dBFS if weighted=False."""
    n = int(duration * sr)
    powers = _block_powers(data[:n], sr, weighted=weighted)
    if len(powers) == 0:
        return -100.0
    mean_power = np.mean(powers)
    return _to_lufs(mean_power) if weighted else _to_dbfs(mean_power)


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


# Backwards-compatible name used elsewhere in this file / by anyone importing it.
def integrated_lufs(data, sr, abs_gate=-70.0):
    return integrated_level(data, sr, abs_gate, weighted=True)


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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_spectrograms(normalized, sr, out_path, chunk_seconds=8, fmax=12000):
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(normalized[names[0]]) // 2

    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    im = None
    for ax, name in zip(axes, names):
        chunk = normalized[name][start:start + chunk_len]
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


def plot_average_spectrum(normalized, sr, out_path, chunk_seconds=8):
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    start = len(normalized[names[0]]) // 2

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load, measure each file's own noise floor from its silent lead-in,
    #    then compute integrated LUFS (adaptively gated relative to that
    #    file's own noise floor, if enabled).
    results = {}
    unit = 'LUFS' if USE_PERCEPTUAL_WEIGHTING else 'dBFS'
    default_gate = -70.0 if USE_PERCEPTUAL_WEIGHTING else -60.0
    for name, p in paths.items():
        sr, data = load(p)
        noise_floor = measure_noise_floor(data, sr, weighted=USE_PERCEPTUAL_WEIGHTING)

        if USE_ADAPTIVE_GATE:
            abs_gate = noise_floor + ADAPTIVE_GATE_MARGIN_DB
        else:
            abs_gate = default_gate

        level = integrated_level(data, sr, abs_gate, weighted=USE_PERCEPTUAL_WEIGHTING)
        results[name] = {'sr': sr, 'data': data, 'lufs': level,
                          'noise_floor': noise_floor, 'abs_gate': abs_gate,
                          'peak': np.max(np.abs(data))}
        print(f"{name}: sr={sr}, samples={len(data)}, "
              f"peak={20*np.log10(results[name]['peak']+1e-12):.2f} dBFS, "
              f"noise floor={noise_floor:.2f} {unit}, gate used={abs_gate:.2f} {unit}, "
              f"integrated={level:.2f} {unit}")

    sr = next(iter(results.values()))['sr']

    # Sanity check: flag if noise floors differ substantially between files --
    # this is the asymmetry we discussed; adaptive gating corrects for it,
    # but it's still worth knowing about.
    floors = [r['noise_floor'] for r in results.values()]
    if max(floors) - min(floors) > 6:
        print(f"\n[note] Noise floors vary by {max(floors)-min(floors):.1f} dB "
              f"across files -- adaptive gating is compensating for this.")

    if TRIM_LEADIN_BEFORE_ANALYSIS:
        n_trim = int(SILENT_LEADIN_SECONDS * sr)
        for r in results.values():
            r['data'] = r['data'][n_trim:]

    # 2. Level-match everything to TARGET_LUFS
    print(f"\n--- Normalizing all files to {TARGET_LUFS} LUFS ---")
    normalized = {}
    for name, r in results.items():
        gain_db = TARGET_LUFS - r['lufs']
        gain_lin = 10 ** (gain_db / 20)
        normalized[name] = r['data'] * gain_lin
        print(f"{name}: gain applied = {gain_db:+.2f} dB, "
              f"new peak = {np.max(np.abs(normalized[name])):.4f}")

    # 3. Null test between the chosen pair
    a_name, b_name = NULL_TEST_PAIR
    print(f"\n--- Null test: {a_name} vs {b_name} (LUFS-matched) ---")
    nt = null_test(normalized[a_name], normalized[b_name], sr)
    print(f"Estimated lag: {nt['lag_samples']} samples ({nt['lag_ms']:.2f} ms)")
    print(f"Residual RMS relative to signal: {nt['residual_pct']:.1f}% "
          f"({nt['residual_db']:.2f} dB)")

    # 4. Plots
    plot_spectrograms(normalized, sr, f'{OUTPUT_DIR}/spectrograms.png')
    plot_average_spectrum(normalized, sr, f'{OUTPUT_DIR}/avg_spectrum.png')
    print(f"\nSaved {OUTPUT_DIR}/spectrograms.png and {OUTPUT_DIR}/avg_spectrum.png")

    # 5. Per-band table (handy for pinpointing exactly where two takes diverge)
    print(f"\n--- Power (dB) by frequency band, referenced to '{list(paths)[0]}' ---")
    print_band_table(normalized, sr, reference=list(paths)[0])

    # optional: stash everything for further ad-hoc analysis
    with open(f'{OUTPUT_DIR}/analysis_cache.pkl', 'wb') as f:
        pickle.dump({'results': results, 'normalized': normalized,
                     'null_test': nt, 'sr': sr}, f)


if __name__ == '__main__':
    main()