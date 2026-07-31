"""
evaluate_prediction.py

Quantitative comparison between a real analog RAT capture and the
model's prediction at the same (d, f, v) settings -- the held-out
interpolation check from the original project plan (v=6 was never
trained on; only v=5 and v=7 were).

Two things have to happen before any sample-level comparison means
anything:
  1. The real capture has its own per-take reamp latency relative to
     the dry reference (same issue diagnosed at length building
     tests/check_alignment.py / black-box/alignment.py) -- the
     prediction has no such latency, since it's generated directly from
     dry with no physical round trip. So the real capture is aligned to
     dry using the same onset-based per-file shift estimator used for
     training data, with the shift estimated from the post-lead-in
     (performance) region, then applied across the FULL file.
  2. Loudness matching (LUFS) before any residual/null-test comparison,
     same convention as spectral_compare.py, so a plain level difference
     doesn't get mistaken for a shape/timbre difference.

Reports:
  - ESR (same formula as training loss) on the aligned, post-lead-in
    region -- directly comparable to training-loss numbers, showing the
    real train/held-out generalization gap rather than a memorization
    number.
  - LUFS-matched null-test residual (%, dB), spectrogram/average-
    spectrum plots, and a per-band table, reusing the exact tooling
    already validated for the RAT's volume-reshapes-tone finding.

Requires infer.py to have been re-run since the lead-in-inclusion fix,
so PRED_PATH covers the full file (including the lead-in) the same way
REAL_PATH does.

Run with: python evaluate_prediction.py
"""
import os, json, sys, torch
import numpy as np
from common.alignment import estimate_shift
from common.cfg import get_config
from black_box.model.model import esr_loss, dc_loss
import eval.spectral_compare as sc
import eval.spectral_compare_db as sc_db

from eval.plot_waves import plot_waveforms

from common.delay_ops import measure_delay, apply_shift
from common.utils import load_wav

config = get_config()


# def align_to_dry(dry_full, other_full, shift):
#     """Slice dry_full/other_full so other_full's content at sample t was
#     recorded `shift` samples after dry_full's content at sample t."""
#     if shift >= 0:
#         other_al = other_full[shift:]
#         dry_al = dry_full[:len(other_al)]
#     else:
#         dry_al = dry_full[-shift:]
#         other_al = other_full[:len(dry_al)]
#     n = min(len(dry_al), len(other_al))
#     return dry_al[:n], other_al[:n]

def evaluate_prediction(
    dry_path = '/Users/owenmeyer/dsp-modeler/model_data/input.wav',
    real_path = '/Users/owenmeyer/dsp-modeler/d-4_f-4_v-6.wav',
    pred_path = '/Users/owenmeyer/dsp-modeler/outputs/pred_d-4_f-4_v-6.wav',
    output_dir = '/Users/owenmeyer/dsp-modeler/outputs',
    target_lufs = -23.0

):
    # sr, dry_full = sc.load(dry_path)
    # sr_r, real_full = sc.load(real_path)
    # sr_p, pred_full = sc.load(pred_path)
    dry_full, sr = load_wav(dry_path)
    real_full, sr_r = load_wav(real_path)
    pred_full, sr_p = load_wav(pred_path)
    assert sr == sr_r == sr_p, "sample rate mismatch between dry/real/pred"

    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sr)

    # 1. Estimate the real capture's per-take latency using onset
    #    detection on the post-lead-in (performance) region only, then
    #    apply that same fixed shift across the FULL files -- the
    #    physical latency is constant for the whole take.
    # shift = estimate_shift(dry_full[n_trim:], real_full[n_trim:], sr)
    shift, sr = measure_delay(
        real_full[n_trim:],
        dry_full[n_trim:],
        sr,
        n_onsets=20, # n strongest moments in the dry guitar recording where a note starts
        n_candidates_per_onset=10,
        window_seconds=0.4,
        search_seconds=1.0,
        preroll_seconds=0.05, # window starts X s before the detected onset
        min_spacing_seconds=3.0, # time between onsets to keep from picking the same note played
        cluster_window_seconds=0.02,
        verbose=False,
    )
    print(shift)
    print(f"real capture shift relative to dry: {shift} samples ({shift/sr*1000:.3f} ms)")

    dry_aligned, real_aligned = apply_shift(dry_full, real_full, shift)
    # dry_aligned, real_aligned = align_to_dry(dry_full, real_full, shift)
    # pred shares dry_full's indexing exactly (no physical round trip),
    # so it gets the same slice dry_aligned came from
    pred_aligned = pred_full[:len(dry_aligned)]

    n = min(len(real_aligned), len(pred_aligned))
    real_aligned, pred_aligned = real_aligned[:n], pred_aligned[:n]

    # 2. restrict to the meaningful (post-lead-in) performance region for
    #    all quantitative comparisons
    dry_eval = dry_aligned[n_trim:]
    real_eval = real_aligned[n_trim:]
    pred_eval = pred_aligned[n_trim:]

    print(f'min {pred_eval.min()}')
    print(f'max {pred_eval.max()}')
    idx = np.argmin(pred_eval)  # or pred_norm, whichever you pulled the stats from
    print(f"global min at t={idx/sr:.2f}s, value={pred_eval[idx]}")

    # --- ESR / DC (same formula as training loss) ---
    dry_t = torch.from_numpy(dry_eval).float().unsqueeze(0).unsqueeze(-1)
    real_t = torch.from_numpy(real_eval).float().unsqueeze(0).unsqueeze(-1)
    pred_t = torch.from_numpy(pred_eval).float().unsqueeze(0).unsqueeze(-1)
    esr = esr_loss(pred_t, real_t).item()
    dc = dc_loss(pred_t, real_t).item()
    print(f"\nESR (held-out, v=6): {esr:.5f}")
    print(f"DC loss: {dc:.6f}")

    # --- LUFS-matched null test (reusing spectral_compare.py's tooling) ---
    real_noise_floor = sc.measure_noise_floor(real_eval, sr)
    pred_noise_floor = sc.measure_noise_floor(pred_eval, sr)
    real_gate = real_noise_floor + 10
    pred_gate = pred_noise_floor + 10
    real_lufs = sc.integrated_level(real_eval, sr, real_gate)
    pred_lufs = sc.integrated_level(pred_eval, sr, pred_gate)

    real_norm = real_eval * 10 ** ((target_lufs - real_lufs) / 20)
    pred_norm = pred_eval * 10 ** ((target_lufs - pred_lufs) / 20)
    dry_norm = dry_eval * 10 ** ((target_lufs - pred_lufs) / 20)

    nt = sc.null_test(real_norm, pred_norm, sr)
    print(f"\nLUFS-matched null test (target {target_lufs} LUFS):")
    print(f"  real: noise_floor={real_noise_floor:.2f} LUFS, integrated={real_lufs:.2f} LUFS")
    print(f"  pred: noise_floor={pred_noise_floor:.2f} LUFS, integrated={pred_lufs:.2f} LUFS")
    print(f"  residual: {nt['residual_pct']:.1f}% ({nt['residual_db']:.2f} dB)")
    print(f"  (extra lag null_test's own internal alignment found: "
          f"{nt['lag_samples']} samples -- should be near zero if our shift was already correct)")

    # --- plots + band table ---
    normalized = {'real_v6': real_norm, 'pred_v6': pred_norm, 'dry_v6': dry_norm}
    os.makedirs(output_dir, exist_ok=True)
    sc.plot_spectrograms(normalized, sr, f'{output_dir}/eval_v6_spectrograms.png')
    sc.plot_average_spectrum(normalized, sr, f'{output_dir}/eval_v6_avg_spectrum.png')
    print(f"\nSaved {output_dir}/eval_v6_spectrograms.png and {output_dir}/eval_v6_avg_spectrum.png")
    sc.print_band_table(normalized, sr, reference='real_v6')

    # plot waves
    waveform_out_path = f'{output_dir}/eval_v6_waveform.png'
    plot_waveforms(normalized, sr, waveform_out_path, chunk_seconds=2, zoom_seconds=0.05)
    print(f"Saved {waveform_out_path}")