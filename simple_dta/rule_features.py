"""Rule-based replacement for the learned protein CNN tower.

Given the position weight matrices (PWMs) fit from a trained model's
most-important channels (see `interpret_protein_tower.py --save-pwm`), score
every window of a raw protein sequence against each motif and keep the max
(a "matched filter" scan -- the hand-coded analogue of what the learned
max-pooled conv channel was doing). One scalar per motif per protein,
computed with plain numpy string/array ops -- no learned protein tower at all.

    PWM: for each position in a fixed-length window, the observed frequency
    of each amino acid across a set of aligned example windows.
    log-odds[pos, aa] = log( (freq[pos, aa] + pseudocount) / background[aa] )
    score(window) = sum_pos log_odds[pos, window[pos]]
"""
import json
import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYBUXZO")  # superset seen in CHARPROTSET

# The 20 standard amino acids -- the only symbols the PWM/scoring alphabet
# uses. B/U/X/Z/O (ambiguous/rare placeholder codes) are treated as "unknown"
# (neutral, score 0) rather than given their own column: their true background
# frequency is near-zero (~1e-6 for 'X' in Davis), so any log-odds ratio built
# against them blows up to huge, meaningless scores from the pseudocount alone.
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")


def background_frequencies(sequences):
    """Per-amino-acid frequency across a list of raw sequences, restricted to
    the 20 standard residues (ambiguous codes excluded, not just downweighted)."""
    counts = {a: 0 for a in STANDARD_AA}
    total = 0
    for s in sequences:
        for ch in s:
            if ch in counts:
                counts[ch] += 1
                total += 1
    return {ch: c / total for ch, c in counts.items()}


def fit_pwm(windows, rf, background, pseudocount=1.0):
    """windows: list of equal-length (rf) aligned strings -> (alphabet, log_odds[rf, |alphabet|]).
    Non-standard residues in a window (rare) simply don't add to any bin at
    that position -- the pseudocount alone still gives them a well-defined
    (small) probability, no different from any other unobserved symbol."""
    alphabet = STANDARD_AA
    idx = {a: i for i, a in enumerate(alphabet)}
    counts = np.full((rf, len(alphabet)), pseudocount, dtype=np.float64)
    for w in windows:
        for pos, ch in enumerate(w):
            if ch in idx:
                counts[pos, idx[ch]] += 1
    freqs = counts / counts.sum(axis=1, keepdims=True)
    bg = np.array([background[a] for a in alphabet], dtype=np.float64)
    bg = bg / bg.sum()
    log_odds = np.log(freqs / bg[None, :])
    return alphabet, log_odds


def load_pwms(path):
    """Load a runs/<tag>_pwm.json file -> list of dicts with a ready-to-use
    char_to_idx map and numpy log_odds matrix, sorted by channel importance
    (delta_MSE desc, matching the order they were saved in)."""
    with open(path) as f:
        raw = json.load(f)
    pwms = []
    for c in raw["channels"]:
        log_odds = np.asarray(c["log_odds"], dtype=np.float64)
        char_to_idx = {a: i for i, a in enumerate(c["alphabet"])}
        pwms.append({"channel": c["channel"], "delta_MSE": c["delta_MSE"],
                    "consensus": c["consensus"], "alphabet": c["alphabet"],
                    "char_to_idx": char_to_idx, "log_odds": log_odds,
                    "rf": log_odds.shape[0]})
    return pwms, raw["receptive_field"]


def _encode_sequence(seq, char_to_idx, unk_idx):
    return np.array([char_to_idx.get(ch, unk_idx) for ch in seq], dtype=np.int64)


def scan_max_score(seq, pwm):
    """Best (max) log-odds score of `pwm` anywhere in `seq`. Unknown/rare
    characters not seen in this PWM's alphabet score 0 (neutral) at that
    position, rather than crashing or being arbitrarily penalized."""
    rf = pwm["rf"]
    if len(seq) < rf:
        return 0.0
    log_odds = pwm["log_odds"]  # [rf, |alphabet|]
    char_to_idx = pwm["char_to_idx"]
    n_alpha = log_odds.shape[1]
    # per-position score lookup table augmented with a neutral (0) column for unknowns
    lut = np.concatenate([log_odds, np.zeros((rf, 1))], axis=1)  # [rf, |alphabet|+1]
    encoded = _encode_sequence(seq, char_to_idx, unk_idx=n_alpha)  # [L]
    windows = np.lib.stride_tricks.sliding_window_view(encoded, rf)  # [n_windows, rf]
    scores = lut[np.arange(rf)[None, :], windows]  # scores[i, pos] = lut[pos, windows[i, pos]]
    return float(scores.sum(axis=1).max())


class RuleFeaturizer:
    """Turns a raw protein sequence into a k-dim vector of max PWM-match
    scores (one per motif channel), caching by sequence so Davis's heavy
    protein reuse (~68 drug pairs/protein on average) only pays the scan
    cost once per unique protein."""

    def __init__(self, pwms):
        self.pwms = pwms
        self.k = len(pwms)
        self._cache = {}

    def __call__(self, seq):
        if seq in self._cache:
            return self._cache[seq]
        feats = np.array([scan_max_score(seq, pwm) for pwm in self.pwms], dtype=np.float32)
        self._cache[seq] = feats
        return feats

    def batch(self, seqs):
        return np.stack([self(s) for s in seqs], axis=0)
