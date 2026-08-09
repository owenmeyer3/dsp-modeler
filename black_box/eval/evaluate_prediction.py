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
from scipy.signal import stft, istft
from scipy.stats import skew

config = get_config()

def asymmetry_report(x, name):
    pos = x[x > 0]
    neg = x[x < 0]
    pos_rms = np.sqrt(np.mean(pos**2)) if len(pos) else 0
    neg_rms = np.sqrt(np.mean(neg**2)) if len(neg) else 0
    p99_9 = np.percentile(x, 99.9)
    p0_1 = np.percentile(x, 0.1)
    print(f"{name}:")
    print(f"  skewness:            {skew(x):+.4f}")
    print(f"  pos_rms/neg_rms:     {pos_rms/neg_rms:.4f}  (1.0 = symmetric)")
    print(f"  99.9th pct / |0.1th pct|: {p99_9/abs(p0_1):.4f}  (1.0 = symmetric)")

def synthesize_noise(noise_profile, sr, n_samples, nperseg=2048):
    # generate white noise, shape its spectrum to match the measured profile
    white = np.random.randn(n_samples)
    f, t, Zxx = stft(white, fs=sr, nperseg=nperseg)
    shaped = Zxx * (noise_profile / (np.abs(Zxx).mean(axis=1, keepdims=True) + 1e-15))
    _, noise = istft(shaped, fs=sr, nperseg=nperseg)
    return noise[:n_samples]


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

    # Get noise only data
    real_noise = real[:n_trim]
    noise_profile = sc.estimate_noise_profile(real_noise, sr)
    synthetic_noise_tile = synthesize_noise(noise_profile, sr, len(real_noise))
    n_tiles = int(np.ceil(len(dry) / len(synthetic_noise_tile)))
    synthetic_noise = np.tile(synthetic_noise_tile, n_tiles)
    
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
    real_noise_eval = real_noise[:n]
    synth_noise_eval = synthetic_noise[:n]
    pred_n_eval = pred_eval + synth_noise_eval

    print(f"real noise RMS:      {np.sqrt(np.mean(real_noise_eval**2)):.6f}")
    print(f"synth noise RMS:      {np.sqrt(np.mean(synth_noise_eval**2)):.6f}")

    print(f'min {pred_eval.min()}')
    print(f'max {pred_eval.max()}')
    idx = np.argmin(pred_eval)  # or pred_norm, whichever you pulled the stats from
    print(f"global min at t={idx/sr:.2f}s, value={pred_eval[idx]}")

    # --- ESR / DC (same formula as training loss) ---
    dry_t = torch.from_numpy(dry_eval).float().unsqueeze(0).unsqueeze(-1)
    real_t = torch.from_numpy(real_eval).float().unsqueeze(0).unsqueeze(-1)
    real_dn_t = torch.from_numpy(real_dn_eval).float().unsqueeze(0).unsqueeze(-1)
    pred_n_t = torch.from_numpy(pred_n_eval).float().unsqueeze(0).unsqueeze(-1)
    pred_t = torch.from_numpy(pred_eval).float().unsqueeze(0).unsqueeze(-1)
    print("\nReal")
    print(f"ESR (held-out, v=6): {esr_loss(pred_t, real_t).item():.5f}")
    print(f"DC loss: {dc_loss(pred_t, real_t).item():.6f}")
    print("\nReal Denoised")
    print(f"ESR (held-out, v=6): {esr_loss(pred_t, real_dn_t).item():.5f}")
    print(f"DC loss: {dc_loss(pred_t, real_dn_t).item():.6f}")
    print("\nSynthetic noise")
    print(f"ESR (held-out, v=6): {esr_loss(pred_n_t, real_t).item():.5f}")
    print(f"DC loss: {dc_loss(pred_n_t, real_t).item():.6f}")
    # --- LUFS-matched null test (reusing spectral_compare.py's tooling) ---
    real_noise_noise_floor = sc.measure_noise_floor(real_noise_eval, sr)
    synth_noise_noise_floor = sc.measure_noise_floor(synth_noise_eval, sr)
    real_noise_gate = real_noise_noise_floor + 10
    synth_noise_gate = synth_noise_noise_floor + 10
    real_noise_lufs = sc.integrated_level(real_noise_eval, sr, real_noise_gate)
    synth_noise_lufs = sc.integrated_level(synth_noise_eval, sr, synth_noise_gate)
    real_noise_norm = real_noise_eval * 10 ** ((target_lufs - real_noise_lufs) / 20)
    synth_noise_norm = synth_noise_eval * 10 ** ((target_lufs - synth_noise_lufs) / 20)

    dry_noise_floor = sc.measure_noise_floor(dry_eval, sr)
    dry_gate = dry_noise_floor + 10
    dry_lufs = sc.integrated_level(dry_eval, sr, dry_gate)
    dry_norm = dry_eval * 10 ** ((target_lufs - dry_lufs) / 20)

    real_noise_floor = sc.measure_noise_floor(real_eval, sr)
    real_dn_noise_floor = sc.measure_noise_floor(real_dn_eval, sr)
    real_gate = real_noise_floor + 10
    real_dn_gate = real_dn_noise_floor + 10
    real_lufs = sc.integrated_level(real_eval, sr, real_gate)
    real_dn_lufs = sc.integrated_level(real_dn_eval, sr, real_dn_gate)
    real_norm = real_eval * 10 ** ((target_lufs - real_lufs) / 20)
    real_dn_norm = real_dn_eval * 10 ** ((target_lufs - real_dn_lufs) / 20)

    pred_n_noise_floor = sc.measure_noise_floor(pred_n_eval, sr)
    pred_noise_floor = sc.measure_noise_floor(pred_eval, sr)
    pred_n_gate = pred_n_noise_floor + 10
    pred_gate = pred_noise_floor + 10
    pred_n_lufs = sc.integrated_level(pred_n_eval, sr, pred_n_gate)
    pred_lufs = sc.integrated_level(pred_eval, sr, pred_gate)
    pred_norm = pred_eval * 10 ** ((target_lufs - pred_lufs) / 20)
    pred_n_norm = pred_eval * 10 ** ((target_lufs - pred_n_lufs) / 20)


    nt = sc.null_test(real_norm, pred_norm, sr)
    print(f"\nLUFS-matched null test (target {target_lufs} LUFS):")
    print(f"  real: noise_floor={real_noise_floor:.2f} LUFS, integrated={real_lufs:.2f} LUFS")
    print(f"  pred: noise_floor={pred_noise_floor:.2f} LUFS, integrated={pred_lufs:.2f} LUFS")
    print(f"  residual: {nt['residual_pct']:.1f}% ({nt['residual_db']:.2f} dB)")
    print(f"  (extra lag null_test's own internal alignment found: "
          f"{nt['lag_samples']} samples -- should be near zero if our shift was already correct)")
  
    nt = sc.null_test(real_dn_norm, pred_norm, sr)
    print(f"\nLUFS-matched null test (target {target_lufs} LUFS):")
    print(f"  real_dn: noise_floor={real_dn_noise_floor:.2f} LUFS, integrated={real_dn_lufs:.2f} LUFS")
    print(f"  pred: noise_floor={pred_noise_floor:.2f} LUFS, integrated={pred_lufs:.2f} LUFS")
    print(f"  residual: {nt['residual_pct']:.1f}% ({nt['residual_db']:.2f} dB)")
    print(f"  (extra lag null_test's own internal alignment found: "
          f"{nt['lag_samples']} samples -- should be near zero if our shift was already correct)")

    nt = sc.null_test(real_norm, pred_n_norm, sr)
    print(f"\nLUFS-matched null test (target {target_lufs} LUFS):")
    print(f"  real: noise_floor={real_noise_floor:.2f} LUFS, integrated={real_lufs:.2f} LUFS")
    print(f"  pred_n: noise_floor={pred_n_noise_floor:.2f} LUFS, integrated={pred_n_lufs:.2f} LUFS")
    print(f"  residual: {nt['residual_pct']:.1f}% ({nt['residual_db']:.2f} dB)")
    print(f"  (extra lag null_test's own internal alignment found: "
          f"{nt['lag_samples']} samples -- should be near zero if our shift was already correct)")
  
    # --- plots + band table ---
    os.makedirs(output_dir, exist_ok=True)

    # serieses = {'real': real_norm, 'real_dn':real_dn_norm, 'pred': pred_norm, 'dry': dry_norm, 'pred_n': pred_n_norm}
    #serieses = {'real': real_norm, 'pred_n': pred_n_norm}
    serieses = {'real_dn': real_dn_norm, 'pred': pred_norm}
    # sc.plot_spectrograms(serieses, sr, f'{output_dir}/eval_norm_spectrograms.png', fmax=24000, vmin=-120)
    # sc.plot_average_spectrum(serieses, sr, f'{output_dir}/eval_norm_avg_spectrum.png')
    # print(f"\nSaved {output_dir}/eval_v6_spectrograms.png and {output_dir}/eval_avg_spectrum.png")
    # sc.print_band_table(serieses, sr, reference='real')

    # --- raw (non-LUFS-matched) dB spectrum, for comparison ---
    # raw_signals = {'real': real_eval, 'real_dn': real_dn_eval, 'pred': pred_eval, 'dry': dry_eval, 'pred_n':pred_n_eval, 'real_noise':real_noise_eval, 'synth_noise':synth_noise_eval}
    raw_signals = {'real': real_eval, 'pred_n':pred_n_eval}
    sc.plot_spectrograms(raw_signals, sr, f'{output_dir}/eval_db_spectrograms.png', fmax=24000, vmin=-120)
    sc_db.plot_average_spectrum(raw_signals, sr, f'{output_dir}/eval_db_avg_spectrum_db.png')
    print(f"Saved {output_dir}/eval_avg_spectrum_db.png")

    # plot waves
    # serieses = {'real_eval': real_eval, 'pred_n_eval': pred_n_eval}
    # serieses = {'real_noise_eval': real_noise_eval, 'synth_noise_eval': synth_noise_eval}
    serieses = {'real_dn_eval': real_dn_eval, 'pred_eval': pred_eval}
    # serieses = {'real_eval': real_eval, 'pred_n_eval': pred_n_eval}

    #serieses = {'real': real_norm, 'pred_n': pred_n_norm}
    # serieses = {'real_dn': real_dn_norm, 'pred': pred_norm}
    # serieses = {'real': real_norm, 'pred_n': pred_n_norm, 'real_dn': real_dn_norm, 'pred': pred_norm}
    # serieses = {'real': real_norm, 'real_dn': real_dn_norm}
    # serieses = {'real': real_norm, 'real_dn': real_dn_norm}
    # serieses = {'real': real_norm, 'real_dn': real_dn_norm, 'pred': pred_norm, 'pred_n': pred_n_norm}
    for k, v in serieses.items():
      asymmetry_report(v, k)
    plot_waveforms(serieses, sr, f'{output_dir}/eval_waveform.png',    chunk_seconds=2, zoom_seconds=0.05, title="half-time window")
    plot_waveforms(serieses, sr, f'{output_dir}/eval_waveform_st.png', chunk_seconds=2, zoom_seconds=0.05, start_seconds=1, title="silence time window")
    print(f"Saved {output_dir}/eval_waveform.png")



    # real_shift_aligned, pred_n_shift_aligned = apply_shift(real_norm, pred_n_norm, nt['lag_samples'])
    # normalized_shift_corrected = {'real': real_shift_aligned, 'pred_n': pred_n_shift_aligned}
    # plot_waveforms(normalized_shift_corrected, sr, f'{output_dir}/eval_waveform_shift_corrected.png', chunk_seconds=2, zoom_seconds=0.05)
    # print(f"Saved {output_dir}/eval_waveform_shift_corrected.png")