"""
alignment.py

Per-wet-file dry/wet sample-shift estimation, used by PedalDataset in
train.py. Each wet take is its own separate reamp (play + record) pass,
and the alignment between dry-out and wet-in can differ slightly between
passes (USB buffer/driver start-up quantization) even though it stays
fixed *within* a single pass -- see tests/check_alignment.py for the
diagnostic history behind this. So the shift is estimated once per file,
not assumed to be one global constant.

Correlation is anchored on detected note onsets rather than arbitrary
timestamps: sustained/held notes are quasi-periodic (many similarly-tall
correlation peaks spaced by the fundamental, so argmax picks between
near-ties at random), and distortion reshapes the waveform rather than
just delaying it (naive "same shape, shifted in time" correlation breaks
down harder the more the pedal is driven). A sharp attack transient is
broadband and non-periodic, so its onset timing survives the pedal's
reshaping even though its harmonic content doesn't.
"""
import numpy as np
from scipy import signal
from common.cfg import get_config


def find_onsets(dry, sr, n_onsets=5, min_spacing_seconds=5.0, block_seconds=0.02):
    """Return sample indices of the n_onsets strongest, well-separated note
    attacks in `dry`, sorted by time."""
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

    top = peak_idx[np.argsort(-onset_strength[peak_idx])][:n_onsets]
    top = np.sort(top)
    return top * block


def find_lag(dry_chunk, wet_search):
    """Cross-correlate dry_chunk against a (larger) wet_search window and
    return the lag (in samples) that best aligns them. Positive lag means
    the wet signal lags behind the dry signal by that many samples."""
    corr = signal.correlate(wet_search, dry_chunk, mode='valid')
    center = (len(wet_search) - len(dry_chunk)) // 2
    return int(np.argmax(np.abs(corr)) - center)


def find_lag_forward(dry_chunk, wet, s, max_lag_samples):
    """Like find_lag, but only searches lag >= 0 (wet lags dry), starting
    the search exactly at `s` rather than centering it. Wet always lags
    dry physically (reamp/pedal/interface round trip), so a forward-only,
    generously-sized search still finds the true delay while halving the
    ambiguity space compared to a symmetric search -- a real help when
    the dry content is sustained/tonal (see estimate_shift's docstring)."""
    wet_search = wet[s:s + len(dry_chunk) + max_lag_samples]
    corr = signal.correlate(wet_search, dry_chunk, mode='valid')
    return int(np.argmax(np.abs(corr)))


def _densest_low_cluster(lags, cluster_window, min_frac):
    """Among sorted candidate lags, return the median of the smallest
    (lowest-lag) group that has at least min_frac of all points within
    cluster_window samples of its low end.

    This matters specifically because a sustained/tonal note's waveform
    repeats every fundamental period, so cross-correlation on such a note
    can lock onto a harmonic multiple of the true delay instead of the
    true delay itself -- never *below* it, only above. Onsets landing on
    a genuine broadband attack (no repeating waveform to alias against)
    still report the true delay correctly. So across enough onsets, the
    true delay shows up as the tightest, *lowest-valued* cluster with
    real support, while harmonic-locked onsets scatter upward into
    higher, sparser bands. Taking the plain median instead would risk
    landing in one of those harmonic bands rather than on the true value.
    """
    lags = np.sort(np.asarray(lags))
    n = len(lags)
    min_count = max(1, int(min_frac * n))
    for i in range(n):
        j = i
        while j < n and lags[j] - lags[i] <= cluster_window:
            j += 1
        if j - i >= min_count:
            return np.median(lags[i:j])
    return np.median(lags)  # no cluster met min_frac -- fall back to plain median


def estimate_shift(dry, wet, sr, n_onsets=20, min_spacing_seconds=3.0,
                    window_seconds=0.4, preroll_seconds=0.05,
                    max_lag_seconds=1.0, cluster_window_seconds=0.0006,
                    cluster_min_frac=0.25):
    """Estimate the fixed sample shift to apply to `wet` (relative to `dry`)
    for this particular file.

    Returns the median lag within the densest, lowest cluster of onset-
    based lag estimates -- robust to two distinct failure modes seen in
    practice: (1) the occasional bad/ambiguous onset (e.g. a soft attack),
    which shows up as an isolated outlier, and (2) sustained/tonal onsets
    whose periodic waveform lets correlation lock onto a harmonic multiple
    of the true delay, which shows up as a *cluster* of onsets scattered
    upward at roughly integer multiples of the true value. A plain median
    across few onsets can land inside one of those harmonic bands rather
    than on the true delay; using many onsets and taking the lowest
    well-supported cluster avoids that. Returns 0 (with a warning) if no
    onset gives a usable measurement, e.g. for a very short or very quiet
    file.
    """
    n = min(len(dry), len(wet))
    dry, wet = dry[:n], wet[:n]

    window_len = int(window_seconds * sr)
    max_lag = int(max_lag_seconds * sr)
    preroll = int(preroll_seconds * sr)

    onsets = find_onsets(dry, sr, n_onsets, min_spacing_seconds)
    onsets = onsets[(onsets - preroll >= 0) &
                     (onsets - preroll + window_len + max_lag <= n)]

    if len(onsets) == 0:
        print("  [alignment] no usable onsets found -- defaulting to shift=0")
        return 0

    lags = []
    for onset in onsets:
        s = onset - preroll
        dry_chunk = dry[s:s + window_len]
        lags.append(find_lag_forward(dry_chunk, wet, s, max_lag))

    cluster_window = int(cluster_window_seconds * sr)
    return int(_densest_low_cluster(lags, cluster_window, cluster_min_frac))


def apply_shift(dry, wet, shift):
    """Slice dry/wet so that wet's content at sample t was recorded shift
    samples after dry's content at sample t (shift > 0, the normal case --
    wet lags dry through the reamp/pedal/interface round trip). Returns
    (dry_aligned, wet_aligned) of equal length."""
    if shift >= 0:
        wet_al = wet[shift:]
        dry_al = dry[:len(wet_al)]
    else:
        dry_al = dry[-shift:]
        wet_al = wet[:len(dry_al)]

    n = min(len(dry_al), len(wet_al))
    return dry_al[:n], wet_al[:n]
