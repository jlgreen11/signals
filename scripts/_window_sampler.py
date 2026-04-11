"""Shared non-overlapping window sampler for all eval scripts.

Addresses SKEPTIC_REVIEW.md § 2 / Tier A2 at the source. Every
evaluation script in `scripts/` previously reimplemented the original
`random.sample(range(min_start, max_start), n_windows)` pattern, which
picks distinct starts but does NOT enforce spacing — the same bug
`scripts/random_window_eval.py` used to have.

This helper replaces all of those with a single rejection-sampling
implementation: starts are drawn from the eligible range with a
minimum pairwise spacing of `window_len` bars (i.e. truly non-overlapping
windows of length `window_len`). If the caller asks for more windows
than the range can support, `n_windows` is clamped down to the maximum
that fits and a warning is printed.

Usage::

    from _window_sampler import draw_non_overlapping_starts

    starts = draw_non_overlapping_starts(
        seed=42,
        min_start=homc_train_window + vol_window + warmup_pad,
        max_start=len(prices) - six_months - 1,
        window_len=six_months,
        n_windows=16,
    )

Returns a sorted list of integer indices; guaranteed that
`abs(starts[i] - starts[j]) >= window_len` for all i != j.
"""

from __future__ import annotations

import random


def draw_non_overlapping_starts(
    *,
    seed: int,
    min_start: int,
    max_start: int,
    window_len: int,
    n_windows: int,
) -> list[int]:
    """Sample `n_windows` window starts with minimum pairwise spacing.

    The guarantee is `abs(starts[i] - starts[j]) >= window_len` for every
    pair — i.e. the windows `[starts[i], starts[i]+window_len)` and
    `[starts[j], starts[j]+window_len)` are genuinely non-overlapping.

    Algorithm: slot-based sampling. The eligible range is divided into
    `max_fit = (max_start - min_start) // window_len` non-overlapping
    slots of size `window_len`, plus a `slack = (max_start - min_start) -
    max_fit * window_len` leftover. We uniformly sample `n_windows`
    distinct slot indices in [0, max_fit), then independently sample a
    per-slot jitter in [0, slack // n_windows] to break the
    deterministic alignment. The final start for slot `k` is
    `min_start + k * window_len + jitter[k]`. Non-overlap is preserved
    because jitter < slack_per_window < window_len.

    Earlier design used pure rejection sampling which was too slow near
    saturation (16 windows out of 19 fittable slots = very low
    acceptance rate). This slot-based version is deterministic in the
    number of RNG draws.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    if min_start >= max_start:
        raise ValueError(
            f"min_start ({min_start}) must be < max_start ({max_start})"
        )

    # Max non-overlapping windows that can fit in the eligible range.
    max_fit = (max_start - min_start) // window_len
    if max_fit < 1:
        raise ValueError(
            f"Cannot fit even one window of length {window_len} in "
            f"[{min_start}, {max_start})"
        )
    if n_windows > max_fit:
        print(
            f"  [WARN] requested n_windows={n_windows} exceeds max "
            f"non-overlapping count {max_fit} for this range; clamping "
            f"to {max_fit}"
        )
        n_windows = max_fit

    rng = random.Random(seed)
    # Sample n_windows distinct slot indices uniformly without replacement.
    # Each slot k maps to start `min_start + k * window_len`. This is
    # guaranteed non-overlapping because consecutive slots are exactly
    # `window_len` apart (the minimum required).
    #
    # Earlier version tried to add per-slot jitter to break the alignment,
    # but any positive jitter can destroy non-overlap between adjacent
    # slots (if slot k jitters forward and slot k+1 doesn't, spacing
    # drops below window_len). Randomness across seeds comes from the
    # combinatorial choice of which slots to occupy — with max_fit=19 and
    # n_windows=16 that's C(19,16) = 969 distinct window sets, which is
    # plenty of variation for multi-seed robustness.
    slot_indices = sorted(rng.sample(range(max_fit), n_windows))

    starts = [min_start + k * window_len for k in slot_indices]

    # Defensive check: verify non-overlap. Always passes by construction.
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            if abs(starts[i] - starts[j]) < window_len:
                raise RuntimeError(
                    f"non-overlap invariant violated: {starts[i]} vs {starts[j]}"
                )

    return starts
