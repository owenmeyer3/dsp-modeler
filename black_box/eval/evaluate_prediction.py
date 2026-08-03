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




def evaluate_prediction(
    dry_path,
    real_path,
    pred_path,
    output_dir,
    target_lufs = -23.0
):
    # wav to data
    dry, sr = load_wav(dry_path)
    real, sr_r = load_wav(real_path)
    pred, sr_p = load_wav(pred_path)
    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sr) # selent sample count
    assert sr == sr_r == sr_p, "sample rate mismatch between dry/real/pred"

    # make noiseless real data
    print(f"dry {len(dry)}")
    noise_profile = sc.estimate_noise_profile(real[:n_trim], sr) # pre-silence
    real_dn = sc.spectral_subtract(real, noise_profile, sr)
    print(f"real_dn {len(real_dn)}")
    

    # Shift real to match dry time-sync
    shift, sr = measure_delay(real[n_trim:], dry[n_trim:], sr, verbose=False) # only post-silence
    dry_aligned, real_aligned = apply_shift(dry, real, shift)
    dry_aligned, real_dn_aligned = apply_shift(dry, real_dn, shift)
    pred_aligned = pred[:len(dry_aligned)]
    print(f"real capture shift relative to dry: {shift} samples ({shift/sr*1000:.3f} ms)")

    # evaluate only length of samples existing after shift
    n = min(len(dry_aligned), len(real_aligned))
    dry_eval = dry_aligned[:n]
    real_eval = real_aligned[:n]
    real_dn_eval = real_dn_aligned[:n]
    pred_eval = pred_aligned[:n]

    print(f'min {pred_eval.min()}')
    print(f'max {pred_eval.max()}')
    idx = np.argmin(pred_eval)  # or pred_norm, whichever you pulled the stats from
    print(f"global min at t={idx/sr:.2f}s, value={pred_eval[idx]}")

    # --- ESR / DC (same formula as training loss) ---
    dry_t = torch.from_numpy(dry_eval).float().unsqueeze(0).unsqueeze(-1)
    real_t = torch.from_numpy(real_eval).float().unsqueeze(0).unsqueeze(-1)
    real_dn_t = torch.from_numpy(real_dn_eval).float().unsqueeze(0).unsqueeze(-1)
    pred_t = torch.from_numpy(pred_eval).float().unsqueeze(0).unsqueeze(-1)
    print("\nReal")
    print(f"ESR (held-out, v=6): {esr_loss(pred_t, real_t).item():.5f}")
    print(f"DC loss: {dc_loss(pred_t, real_t).item():.6f}")
    print("\nReal Denoised")
    print(f"ESR (held-out, v=6): {esr_loss(pred_t, real_dn_t).item():.5f}")
    print(f"DC loss: {dc_loss(pred_t, real_dn_t).item():.6f}")

    # --- LUFS-matched null test (reusing spectral_compare.py's tooling) ---
    real_noise_floor = sc.measure_noise_floor(real_eval, sr)
    real_dn_noise_floor = sc.measure_noise_floor(real_dn_eval, sr)
    pred_noise_floor = sc.measure_noise_floor(pred_eval, sr)
    dry_noise_floor = sc.measure_noise_floor(dry_eval, sr)

    real_gate = real_noise_floor + 10
    real_dn_gate = real_dn_noise_floor + 10
    pred_gate = pred_noise_floor + 10
    dry_gate = dry_noise_floor + 10

    real_lufs = sc.integrated_level(real_eval, sr, real_gate)
    real_dn_lufs = sc.integrated_level(real_dn_eval, sr, real_dn_gate)
    pred_lufs = sc.integrated_level(pred_eval, sr, pred_gate)
    dry_lufs = sc.integrated_level(dry_eval, sr, dry_gate)

    real_norm = real_eval * 10 ** ((target_lufs - real_lufs) / 20)
    real_dn_norm = real_dn_eval * 10 ** ((target_lufs - real_dn_lufs) / 20)
    pred_norm = pred_eval * 10 ** ((target_lufs - pred_lufs) / 20)
    dry_norm = dry_eval * 10 ** ((target_lufs - dry_lufs) / 20)

    # nt = sc.null_test(real_norm, pred_norm, sr)
    # print(f"\nLUFS-matched null test (target {target_lufs} LUFS):")
    # print(f"  real: noise_floor={real_noise_floor:.2f} LUFS, integrated={real_lufs:.2f} LUFS")
    # print(f"  pred: noise_floor={pred_noise_floor:.2f} LUFS, integrated={pred_lufs:.2f} LUFS")
    # print(f"  residual: {nt['residual_pct']:.1f}% ({nt['residual_db']:.2f} dB)")
    # print(f"  (extra lag null_test's own internal alignment found: "
    #       f"{nt['lag_samples']} samples -- should be near zero if our shift was already correct)")

    # --- plots + band table ---
    os.makedirs(output_dir, exist_ok=True)

    normalized = {'real': real_norm, 'real_dn':real_dn_norm, 'pred': pred_norm, 'dry': dry_norm}
    sc.plot_spectrograms(normalized, sr, f'{output_dir}/eval_norm_spectrograms.png', fmax=24000, vmin=-120)
    sc.plot_average_spectrum(normalized, sr, f'{output_dir}/eval_norm_avg_spectrum.png')
    print(f"\nSaved {output_dir}/eval_v6_spectrograms.png and {output_dir}/eval_avg_spectrum.png")
    sc.print_band_table(normalized, sr, reference='real')

    # --- raw (non-LUFS-matched) dB spectrum, for comparison ---
    raw_signals = {'real': real_eval, 'real_dn': real_dn_eval, 'pred': pred_eval, 'dry': dry_eval}
    sc.plot_spectrograms(raw_signals, sr, f'{output_dir}/eval_db_spectrograms.png', fmax=24000, vmin=-120)
    sc_db.plot_average_spectrum(raw_signals, sr, f'{output_dir}/eval_db_avg_spectrum_db.png')
    print(f"Saved {output_dir}/eval_avg_spectrum_db.png")


    # plot waves
    waveform_out_path = f'{output_dir}/eval_waveform.png'
    plot_waveforms(normalized, sr, waveform_out_path, chunk_seconds=2, zoom_seconds=0.05)
    print(f"Saved {waveform_out_path}")