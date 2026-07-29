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
# Loading
# ---------------------------------------------------------------------------
def load(path):
    """Read a wav file, return (sample_rate, mono float array in [-1, 1])."""
    sr, raw = wavfile.read(path)
    data = raw.astype(np.float64)

    # Integer PCM (16/24/32-bit) needs scaling down to the [-1, 1] range.
    # scipy stores samples left-justified in the container dtype, so
    # dividing by that dtype's own max magnitude gets us back to [-1, 1].
    if np.issubdtype(raw.dtype, np.integer):
        max_val = float(2 ** (raw.dtype.itemsize * 8 - 1))
        data = data / max_val

    if data.ndim > 1:            # stereo -> mono
        data = data.mean(axis=1)

    return sr, data


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


def integrated_dbfs(data, sr, gate_db):
    """The overall level of the *played* content, ignoring silence.

    Why this needs a gate at all: if we just averaged the power of the
    entire file including silence, a file with lots of silent gaps would
    come out quieter than one with less silence, even if the actual
    playing was identical. Gating throws out blocks that are basically
    just noise/silence, so the result reflects "how loud is this when
    something is actually happening."

    gate_db: any block quieter than this is treated as silence
             and excluded from the average. Pass in something based on
             THIS file's own measured noise floor (see main() below) so
             the threshold makes sense regardless of how hot or quiet
             the file was originally recorded.
    """
    powers = block_powers(data, sr)
    levels = power_to_db(powers)

    kept = powers[levels > gate_db]
    if len(kept) == 0:
        # Nothing passed the gate -- the whole file is quieter than the
        # gate itself, so just fall back to using everything.
        kept = powers

    return power_to_db(np.mean(kept))


# ---------------------------------------------------------------------------
# Null test: how similar are two signals once their levels are matched?
# ---------------------------------------------------------------------------
def null_test(a, b, sr, chunk_seconds=5, search_seconds=1):
    """Subtracts one signal from the other and measures what's left over. 
    If two signals are truly identical, the residual after subtraction is silence. 
    The louder the residual relative to the original signal, the more the two signals
    actually differ, independent of how loud each one is."""

    corr = signal.correlate(b, a, mode='full')
    lag = np.argmax(np.abs(corr)) - (len(a) - 1)  

    if lag >= 0:
        a_aligned = a[:len(a) - lag]
        b_aligned = b[lag:lag + len(a_aligned)]
    else:
        b_aligned = b[:len(b) + lag]
        a_aligned = a[-lag:-lag + len(b_aligned)]

    n = min(len(a_aligned), len(b_aligned))
    a_aligned, b_aligned = a_aligned[:n], b_aligned[:n]

    residual = a_aligned - b_aligned
    rms_signal = np.sqrt(np.mean(a_aligned ** 2))
    rms_residual = np.sqrt(np.mean(residual ** 2))

    return {
        'lag_samples': lag,
        'lag_ms': lag / sr * 1000,
        'residual_pct': 100 * rms_residual / rms_signal,
        'residual_db': 20 * np.log10(rms_residual / rms_signal),
    }


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

def plot_waveforms(signals, sr, out_path, downsample_to=5000):
    """Plot full-length waveforms of multiple signals stacked vertically,
    sharing a time axis, so you can visually compare their overall shape,
    dynamics, and timing across the whole file.

    downsample_to: target number of points per signal in the final plot.
    Full-resolution plotting of a multi-minute file would be extremely
    slow and mostly invisible anyway, so instead we chop the signal into
    `downsample_to` chunks and plot the min/max envelope of each chunk --
    this preserves the visual peaks (like a DAW's waveform view) while
    being fast to render.
    """
    names = list(signals.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 2.5 * len(names)),
                              sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = signals[name]
        n = len(data)
        chunk_size = max(1, n // downsample_to)
        n_chunks = n // chunk_size

        trimmed = data[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
        mins = trimmed.min(axis=1)
        maxs = trimmed.max(axis=1)
        t = np.arange(n_chunks) * chunk_size / sr

        ax.fill_between(t, mins, maxs, linewidth=0, color='steelblue')
        ax.set_ylabel(name)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Full-length waveform comparison (level-matched)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def save_normalized_wavs(signals, sr, out_dir):
    """Write each (already gain-adjusted) signal to its own .wav file
    in out_dir, so you can listen to or reuse the level-matched versions
    outside this script."""
    os.makedirs(out_dir, exist_ok=True)
    for name, data in signals.items():
        # Guard against any sample that crept past full scale -- writing
        # values outside [-1, 1] would wrap around/distort on export.
        clipped = np.clip(data, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)
        out_path = os.path.join(out_dir, f'{name}_normalized.wav')
        wavfile.write(out_path, sr, int16_data)
        print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Step 1: load every file, and for each one measure:
    #   - its own noise floor (from the known-silent lead-in)
    #   - its gate threshold (noise floor + margin, see integrated_dbfs docstring)
    #   - its overall played-content level (gated average, in dB)
    results = {}
    srs = []
    # Each file
    for name, p in paths.items():
        sr, data = load(p)
        srs.append(sr) # collect sample rates of each track to check for consistency
        noise_floor = measure_noise_floor(data, sr) # get NOISE FLOOR AVG based on start silence
        gate = noise_floor + ADAPTIVE_GATE_MARGIN_DB # Manual ADJUST noise floor average to include silent blocks exceeding the average
        level = integrated_dbfs(data, sr, gate) # get OVERALL LEVEL of the *played* content, ignoring silence. 

        results[name] = {
            'data': data, 
            'level': level, 
            'noise_floor': noise_floor
        }
        print(f"{name}: noise floor = {noise_floor:.2f} dB, gate = {gate:.2f} dB, integrated level = {level:.2f} dB")

    # Get sample rate of files
    assert len(set(srs)) == 1, 'cannot compare files with different sample rates'
    sr = srs[0]

    # Step 2: 
    # - level-match everything to the same target, so any differences we see afterward are about *shape*, not loudness.
    # - This is just multiplying every sample by one constant number -- 
    #   gain_db decides how many dB to shift, 
    #   gain_lin is that same shift converted into a plain multiplier (10^(dB/20) is the standard dB-to-linear-amplitude conversion).
    print(f"\n--- Normalizing everything to {TARGET_DBFS} dB ---")
    normalized = {}
    for name, r in results.items():
        gain_db = TARGET_DBFS - r['level'] # Get track gain delta to target
        gain_lin = 10 ** (gain_db / 20)    # db gain delta to amplitude delta
        normalized[name] = r['data'] * gain_lin # normalize track data samples
        print(f"{name}: applied {gain_db:+.2f} dB (new peak = {np.max(np.abs(normalized[name])):.4f})")

    if TRIM_LEADIN_BEFORE_ANALYSIS:
        n_trim = int(SILENT_LEADIN_SECONDS * sr)
        normalized = {name: sig[n_trim:] for name, sig in normalized.items()}

    # Step 3: null test between the two takes you care about comparing.
    a_name, b_name = NULL_TEST_PAIR
    print(f"\n--- Null test: {a_name} vs {b_name} ---")
    nt = null_test(normalized[a_name], normalized[b_name], sr)
    print(f"Lag: {nt['lag_ms']:.2f} ms | Residual: {nt['residual_pct']:.1f}% of signal ({nt['residual_db']:.2f} dB)")

    # Step 4: visualize.
    plot_spectrograms(normalized, sr, f'{OUTPUT_DIR}/spectrograms_dbfs.png')
    plot_average_spectrum(normalized, sr, f'{OUTPUT_DIR}/avg_spectrum_dbfs.png')
    plot_waveforms(normalized, sr, f'{OUTPUT_DIR}/waveforms.png')
    save_normalized_wavs(normalized, sr, os.path.join(OUTPUT_DIR, 'normalized_wavs')) # not for degital use, a downsampled human listenable comparison

    print(f"\nSaved {OUTPUT_DIR}/spectrograms_dbfs.png and {OUTPUT_DIR}/avg_spectrum_dbfs.png")

    with open(f'{OUTPUT_DIR}/analysis_cache_dbfs.pkl', 'wb') as f:
        pickle.dump({'results': results, 'normalized': normalized, 'null_test': nt}, f)


if __name__ == '__main__':
    main()