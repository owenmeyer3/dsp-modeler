"""
check_alignment.py

PedalDataset (train.py) pairs input.wav against every wet take index-for-index
with NO lag correction -- it just trims the silent lead-in from both and
assumes they're already sample-aligned (comment: "Dry and wet should already
be the same length/aligned ... if a take got trimmed differently, this just
clips to the shorter one"). That assumption has never actually been checked
against the real model_data/ files -- it's only been verified in the separate
analysis scripts (spectral_compare.py etc.), on different files.

This script checks it directly: cross-correlate dry vs each wet take at
several points spread across the file (not just one spot), so a genuine
lag AND clock drift (lag changing over time) both get caught. A nonzero,
*consistent* lag would just mean PedalDataset needs a fixed sample-shift
correction. A lag that *drifts* between windows would mean something worse
(the interface's two channels aren't running off the same clock) and the
per-file "clip to shorter" trick would be pairing wrong samples throughout
training, which would explain the model plateauing at ESR~1 (~= it's not
predicting the target because it's misaligned with it, not because it
hasn't learned yet).

Correlation windows are anchored on detected note onsets rather than
evenly-spaced timestamps. Two reasons arbitrary timestamps gave garbage
(erratic, sign-flipping lag that got *worse* with a wider search range,
not better):
  1. Sustained/held notes are quasi-periodic, so a window landing on one
     has many similarly-tall correlation peaks spaced by the fundamental
     period -- argmax picks between near-ties essentially at random, and
     widening the search window just exposes more decoy peaks.
  2. Distortion reshapes the waveform (flattened peaks, added harmonics,
     different zero-crossings), not just delays it -- naive time-domain
     correlation assumes "same shape, shifted in time," which breaks down
     the harder the pedal is driven (worse for high distortion settings).
A sharp pick attack from near-silence is broadband and non-periodic, so
it correlates far more reliably, and its onset *timing* survives the
pedal's distortion even though its harmonic content doesn't.

Run with: python check_alignment.py
"""
import os, json, glob
import numpy as np
from scipy.io import wavfile
from scipy import signal
from common.cfg import get_config

# ---------------------------------------------------------------------------
# CONFIG -- matches train.py's actual settings
# ---------------------------------------------------------------------------
DATA_DIR = '/Users/owenmeyer/dsp-modeler/model_data'
DRY_FILENAME = 'input.wav'

N_ONSETS = 5             # how many of the strongest, well-separated onsets to check
ONSET_MIN_SPACING_SECONDS = 5.0   # don't pick two onsets from the same note/phrase
ONSET_ENVELOPE_BLOCK = 0.02       # 20ms RMS envelope block, for onset detection only
ONSET_PREROLL_SECONDS = 0.05      # start the correlation window slightly before
                                  # the detected onset, in case detection is a
                                  # few ms late (envelope smoothing lags the true attack)
WINDOW_SECONDS = 0.4     # dry chunk length anchored on each onset
SEARCH_SECONDS = 2.0     # +/- range to search in the wet signal for a match

# Anything beyond this is worth a fixed-offset fix in PedalDataset.
FLAG_LAG_SAMPLES = 5
# If lag varies by more than this across onsets, that's drift, not a fixed offset.
FLAG_DRIFT_SAMPLES = 5


config = get_config(f)


def find_onsets(dry, sr, n_onsets, min_spacing_seconds, block_seconds):
    """Return sample indices of the n_onsets strongest, well-separated note
    attacks in `dry`, sorted by time. Uses a plain energy envelope rather
    than anything spectral -- a pick attack's sharp broadband rise is easy
    to find this way, and we don't need onset *classification*, just a few
    good reliable timing anchors."""
    block = max(1, int(block_seconds * sr))
    n_blocks = len(dry) // block
    envelope_db = np.empty(n_blocks)
    for i in range(n_blocks):
        chunk = dry[i * block:(i + 1) * block]
        rms = np.sqrt(np.mean(chunk ** 2))
        envelope_db[i] = 20 * np.log10(rms + 1e-12)

    onset_strength = np.diff(envelope_db, prepend=envelope_db[0])
    onset_strength[onset_strength < 0] = 0

    min_spacing_blocks = max(1, int(min_spacing_seconds / block_seconds))
    peak_idx, _ = signal.find_peaks(onset_strength, distance=min_spacing_blocks)

    if len(peak_idx) == 0:
        return np.array([], dtype=int)

    # keep the strongest n_onsets, then present them in time order
    top = peak_idx[np.argsort(-onset_strength[peak_idx])][:n_onsets]
    top = np.sort(top)
    return top * block


def load(path):
    """Load a wav file, return (sample_rate, mono float64 array in [-1, 1])."""
    sr, raw = wavfile.read(path)
    data = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        data = data / float(2 ** (raw.dtype.itemsize * 8 - 1))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def find_lag(dry_chunk, wet_search):
    """Cross-correlate dry_chunk against a (larger) wet_search window and
    return the lag (in samples) that best aligns them. Positive lag means
    the wet signal lags behind the dry signal by that many samples."""
    corr = signal.correlate(wet_search, dry_chunk, mode='valid')
    center = (len(wet_search) - len(dry_chunk)) // 2
    return int(np.argmax(np.abs(corr)) - center)


def check_file(dry, wet, sr, label):
    n = min(len(dry), len(wet))
    dry, wet = dry[:n], wet[:n]

    window_len = int(WINDOW_SECONDS * sr)
    search_len = int(SEARCH_SECONDS * sr)
    preroll = int(ONSET_PREROLL_SECONDS * sr)

    onsets = find_onsets(dry, sr, N_ONSETS, ONSET_MIN_SPACING_SECONDS, ONSET_ENVELOPE_BLOCK)
    # keep only onsets with enough room on both sides for the window/search
    onsets = onsets[(onsets - preroll - search_len >= 0) &
                     (onsets - preroll + window_len + search_len <= n)]

    if len(onsets) == 0:
        print(f"{label}: no usable onsets found (file too short, or too quiet "
              f"throughout to detect a note attack)")
        return

    lags = []
    print(f"\n{label} (n={n} samples, {n/sr:.1f}s after lead-in trim, "
          f"{len(onsets)} onset(s) found)")
    for onset in onsets:
        s = onset - preroll
        dry_chunk = dry[s:s + window_len]
        wet_search = wet[s - search_len:s + window_len + search_len]
        lag = find_lag(dry_chunk, wet_search)
        lags.append(lag)
        t = s / sr
        print(f"  onset t={t:6.2f}s: lag = {lag:+d} samples ({lag/sr*1000:+.3f} ms)")

    lags = np.array(lags)
    drift = lags.max() - lags.min()
    print(f"  -> lag range: [{lags.min()}, {lags.max()}] samples, drift = {drift} samples")

    if np.abs(lags).max() > FLAG_LAG_SAMPLES:
        print(f"  [FLAG] consistent nonzero lag > {FLAG_LAG_SAMPLES} samples -- "
              f"PedalDataset needs a fixed sample-shift correction for this file")
    if drift > FLAG_DRIFT_SAMPLES:
        print(f"  [FLAG] lag drifts by {drift} samples across the file -- "
              f"dry/wet are not on a consistent clock; index-for-index pairing "
              f"will progressively misalign, no single fixed shift will fix this")
    if np.abs(lags).max() <= FLAG_LAG_SAMPLES and drift <= FLAG_DRIFT_SAMPLES:
        print(f"  [OK] alignment looks clean (within +/-{FLAG_LAG_SAMPLES} samples, "
              f"no meaningful drift)")


def main():
    dry_path = os.path.join(DATA_DIR, DRY_FILENAME)
    sr, dry_full = load(dry_path)
    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sr)
    dry_full = dry_full[n_trim:]

    all_wavs = sorted(glob.glob(os.path.join(DATA_DIR, '*.wav')))
    wet_paths = [p for p in all_wavs if os.path.basename(p) != DRY_FILENAME]

    print(f"Dry reference: {DRY_FILENAME} (sr={sr})")
    print(f"Checking {len(wet_paths)} wet take(s), up to {N_ONSETS} onset(s) each, "
          f"{WINDOW_SECONDS}s per window, +/-{SEARCH_SECONDS}s search range")

    for wet_path in wet_paths:
        wet_sr, wet_full = load(wet_path)
        assert wet_sr == sr, f"{wet_path} has a different sample rate than {DRY_FILENAME}"
        wet_full = wet_full[n_trim:]
        check_file(dry_full, wet_full, sr, os.path.basename(wet_path))


if __name__ == '__main__':
    main()
