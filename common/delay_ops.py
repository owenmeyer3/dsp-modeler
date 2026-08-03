"""
delay_ops.py

Standalone dry/wet delay measurement. No imports from anywhere else in
this repo -- just numpy/scipy -- so it can be copied out or run on its
own.

Approach: cross-correlate windows anchored on the dry signal's sharpest
note onsets (broadband, non-periodic attacks correlate far more reliably
than sustained/tonal content, which is quasi-periodic and lets
correlation lock onto a harmonic multiple of the true delay instead of
the true delay itself). For each onset, keep the top-N correlation
peaks ("bins"), not just the single best one, then pick whichever lag
value has support from the most *distinct onsets* across the whole
file -- a single bad/ambiguous onset can produce several internally
self-consistent but wrong candidates, and counting raw candidates
instead of distinct onsets lets that one bad onset outvote several
onsets that only weakly agree with each other.
"""
import numpy as np
from scipy import signal
from common.utils import load_wav

def _find_onsets(dry, sample_rate, n_onsets, min_spacing_seconds, block_seconds=0.02):
    """Return sample indices of the n_onsets strongest, well-separated note
    attacks in `dry`, sorted by time."""
    block = max(1, int(block_seconds * sample_rate))
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
    return np.sort(top) * block


def _find_lag_candidates(dry_chunk, wet_search, n_candidates=10, min_spacing_samples=50):
    """Cross-correlate dry_chunk against a (larger) wet_search window and
    return the top n_candidates peaks, sorted by strength -- not just the
    single best match -- so a caller can see whether the winning lag is an
    isolated clear peak or sits among a cluster of similarly-strong,
    harmonically-spaced decoys. Strength is normalized by chunk/search
    energy so it's comparable across onsets of different loudness.
    Positive lag means wet lags behind dry by that many samples."""
    corr = signal.correlate(wet_search, dry_chunk, mode='valid')
    center = (len(wet_search) - len(dry_chunk)) // 2
    abs_corr = np.abs(corr)

    norm = np.sqrt(np.sum(dry_chunk ** 2) * np.sum(wet_search ** 2))
    strength = abs_corr / (norm + 1e-12)

    peak_idx, _ = signal.find_peaks(strength, distance=min_spacing_samples)
    if len(peak_idx) == 0:
        return []

    order = np.argsort(-strength[peak_idx])[:n_candidates]
    top = peak_idx[order]
    lags = (top - center).astype(int)
    return sorted(zip(lags.tolist(), strength[top].tolist()), key=lambda p: -p[1])


def cluster_by_onset_agreement(all_candidates, cluster_window):
    """Pick the lag band supported by the most *distinct onsets*, not the
    most raw candidates. all_candidates is a list of (lag, strength,
    onset_time_sec). Returns (lag, n_onsets_support, members) where
    members is the list of candidates in the winning band."""
    order = np.argsort([c[0] for c in all_candidates])
    sorted_c = [all_candidates[i] for i in order]
    lags = np.array([c[0] for c in sorted_c])

    best = None  # (n_onsets, total_strength, i, j)
    n = len(lags)
    for i in range(n):
        j = i
        while j < n and lags[j] - lags[i] <= cluster_window:
            j += 1
        window = sorted_c[i:j]
        onsets_here = set(c[2] for c in window)
        # best (max) strength contributed per onset, summed -- so one
        # onset can't pad its own score with many near-duplicate peaks
        strength = sum(max(c[1] for c in window if c[2] == t) for t in onsets_here)
        key = (len(onsets_here), strength)
        if best is None or key > best[:2]:
            best = (len(onsets_here), strength, i, j)

    n_onsets_support, _, i, j = best
    cluster_lag = float(np.median(lags[i:j]))
    return cluster_lag, n_onsets_support, sorted_c[i:j]
    
def measure_delay(
    wet_data,
    dry_data,
    sample_rate,
    n_onsets=20, # n strongest moments in the dry guitar recording where a note starts
    n_candidates_per_onset=10,
    window_seconds=0.4,
    search_seconds=1.0,
    preroll_seconds=0.05, # window starts X s before the detected onset
    min_spacing_seconds=3.0, # time between onsets to keep from picking the same note played
    cluster_window_seconds=0.02,
    verbose=False,
):
    """Measure the sample delay of wet_path relative to dry_path 
    positive = wet lags dry
    Returns (delay_samples, sample_rate).

    search_seconds bounds how far the delay search looks -- must exceed
    whatever the real dry/wet delay can be for your setup, or the true
    answer is structurally unreachable no matter how much data you have.

    cluster_window_seconds should be loosened past whatever onset-to-onset
    jitter your setup/detector actually produces; too tight and genuine
    agreement gets fragmented into separate bins.
    """
    assert len(wet_data) == len(dry_data), f"wet/dry data must be equal length"

    n = min(len(dry_data), len(wet_data))
    dry, wet = dry_data[:n], wet_data[:n]

    window_len = int(window_seconds * sample_rate)
    search_len = int(search_seconds * sample_rate)
    preroll = int(preroll_seconds * sample_rate)
    cluster_window = int(cluster_window_seconds * sample_rate)

    onsets = _find_onsets(dry, sample_rate, n_onsets, min_spacing_seconds)
    onsets = onsets[(onsets - preroll - search_len >= 0) &
                     (onsets - preroll + window_len + search_len <= n)]
    if len(onsets) == 0:
        raise RuntimeError("no usable onsets found -- file too short, too quiet, "
                            "or search_seconds/window_seconds too large for it")

    all_candidates = []
    for onset in onsets:
        s = onset - preroll
        dry_chunk = dry[s:s + window_len]
        wet_search = wet[s - search_len:s + window_len + search_len]
        candidates = _find_lag_candidates(dry_chunk, wet_search, n_candidates_per_onset)
        t = s / sample_rate
        if verbose:
            if candidates:
                bins = ", ".join(f"{lag:+d}({lag / sample_rate * 1000:+.2f}ms)@{st:.3f}" for lag, st in candidates)
                print(f"  onset t={t:6.2f}s bins: {bins}")
            else:
                print(f"  onset t={t:6.2f}s: no correlation peak found")
        all_candidates.extend((lag, st, t) for lag, st in candidates)

    if not all_candidates:
        raise RuntimeError("no correlation peaks found across any onset")

    delay_samples, n_onsets_support, members = cluster_by_onset_agreement(all_candidates, cluster_window)

    if verbose:
        support_onsets = sorted(set(round(c[2], 2) for c in members))
        print(f"-> delay: {delay_samples:+.0f} samples ({delay_samples / sample_rate * 1000:+.3f} ms), "
              f"supported by {len(members)}/{len(all_candidates)} candidates "
              f"from {n_onsets_support}/{len(onsets)} onset(s): {support_onsets}")

    print(f"\n{delay_samples:+.0f} samples ({delay_samples / sample_rate * 1000:+.3f} ms)")
    return int(delay_samples), sample_rate

def apply_shift(dry_data, wet_data, delay_samples):
    """Slice dry_data/wet_data so that wet's content at sample t was recorded shift
    samples after dry's content at sample t (delay_samples > 0, the normal case --
    wet lags dry through the reamp/pedal/interface round trip). Returns
    (dry_aligned, wet_aligned) of equal length."""
    print(delay_samples)
    if delay_samples >= 0:
        wet_al = wet_data[delay_samples:]
        dry_al = dry_data[:len(wet_al)]
    else:
        dry_al = dry_data[-delay_samples:]
        wet_al = wet_data[:len(dry_al)]

    n = min(len(dry_al), len(wet_al))
    return dry_al[:n], wet_al[:n]