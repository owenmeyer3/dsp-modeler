"""
plot_waveform_comparison.py

Time-domain waveform comparison to sit alongside evaluate_prediction.py's
eval_v6_spectrograms.png / eval_v6_avg_spectrum.png -- those show frequency
content, this shows the actual sample-by-sample shape, which is where
things like timing offsets, clipping, or envelope/amplitude mismatches
are easiest to spot by eye.

Same alignment + LUFS-matching convention as evaluate_prediction.py:
real capture has its own per-take reamp latency relative to dry (the
prediction doesn't, since it's generated directly from dry with no
physical round trip), so real is aligned to dry first, then both are
level-matched before plotting so a plain loudness difference doesn't
get mistaken for a shape difference.

Run with: python plot_waveform_comparison.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI needed; saves directly to file
import matplotlib.pyplot as plt

from common.delay_ops import measure_delay, apply_shift
from common.cfg import get_config
import eval.spectral_compare as sc

config = get_config()


def plot_waveforms(normalized, sr, out_path, chunk_seconds=2, zoom_seconds=0.05):
    """Stacked full-chunk waveform per signal, plus a short zoomed-in
    overlay panel so shape differences (not just separate, hard-to-compare
    plots) are visible directly on the same axes."""
    names = list(normalized.keys())
    chunk_len = int(chunk_seconds * sr)
    zoom_len = int(zoom_seconds * sr)
    start = len(normalized[names[0]]) // 2

    t_full = np.arange(chunk_len) / sr
    t_zoom = np.arange(zoom_len) / sr * 1000  # ms

    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(12, 2.2 * len(names) + 3))

    for ax, name in zip(axes[:-1], names):
        chunk = normalized[name][start:start + chunk_len]
        ax.plot(t_full, chunk, linewidth=0.5)
        ax.set_ylabel('Amplitude')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel('Time (s)')

    ax = axes[-1]
    for name in names:
        zoom_chunk = normalized[name][start:start + zoom_len]
        ax.plot(t_zoom, zoom_chunk, label=name, linewidth=1.0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Zoomed overlay ({zoom_seconds * 1000:.0f}ms, both signals on the same axes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main(
    dry_path='/Users/owenmeyer/dsp-modeler/model_data/input.wav',
    real_path='/Users/owenmeyer/dsp-modeler/d-4_f-4_v-6.wav',
    pred_path='/Users/owenmeyer/dsp-modeler/outputs/pred_d-4_f-4_v-6.wav',
    output_dir='/Users/owenmeyer/dsp-modeler/outputs',
    target_lufs=-23.0,
):
    sr, dry_full = sc.load(dry_path)
    sr_r, real_full = sc.load(real_path)
    sr_p, pred_full = sc.load(pred_path)
    assert sr == sr_r == sr_p, "sample rate mismatch between dry/real/pred"

    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sr)
    dry_trim = dry_full[n_trim:]
    real_trim = real_full[n_trim:]
    n_common = min(len(dry_trim), len(real_trim))
    dry_trim, real_trim = dry_trim[:n_common], real_trim[:n_common]

    delay_samples, sr = measure_delay(real_trim, dry_trim, sr, n_onsets=20, search_seconds=1.0, cluster_window_seconds=0.02)
    print(f"real capture shift relative to dry: {delay_samples} samples ({delay_samples / sr * 1000:.3f} ms)")

    dry_aligned, real_aligned = apply_shift(dry_full, real_full, delay_samples)
    # pred shares dry_full's indexing exactly (no physical round trip),
    # so it gets the same slice dry_aligned came from
    pred_aligned = pred_full[:len(dry_aligned)]

    n = min(len(real_aligned), len(pred_aligned))
    real_aligned, pred_aligned = real_aligned[:n], pred_aligned[:n]

    real_eval = real_aligned[n_trim:]
    pred_eval = pred_aligned[n_trim:]

    real_noise_floor = sc.measure_noise_floor(real_eval, sr)
    pred_noise_floor = sc.measure_noise_floor(pred_eval, sr)
    real_gate = real_noise_floor + 10
    pred_gate = pred_noise_floor + 10
    real_lufs = sc.integrated_level(real_eval, sr, real_gate)
    pred_lufs = sc.integrated_level(pred_eval, sr, pred_gate)

    real_norm = real_eval * 10 ** ((target_lufs - real_lufs) / 20)
    pred_norm = pred_eval * 10 ** ((target_lufs - pred_lufs) / 20)

    normalized = {'real_v6': real_norm, 'pred_v6': pred_norm}
    os.makedirs(output_dir, exist_ok=True)
    out_path = f'{output_dir}/eval_v6_waveform.png'
    plot_waveforms(normalized, sr, out_path)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
