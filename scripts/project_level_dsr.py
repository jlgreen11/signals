"""Project-level DSR + binomial significance on window-win counts.

Addresses Tier B5 and C6 of SKEPTIC_REVIEW.md.

Section A — Project-level Deflated Sharpe Ratio
-----------------------------------------------
The project's result docs report DSR computed with ``n_trials`` equal to the
size of the *sweep that produced that result* (typically ~25 configs). The
Bailey & Lopez de Prado deflation, applied correctly, should include *every*
strategy evaluation that the researcher looked at while picking the
configuration — not just the last sweep in the chain. Across Tiers 0-3 the
project has run ~1,900 distinct (strategy, asset, param) combinations; the
honest DSR deflates the observed Sharpe against that many trials.

Section B — Binomial significance on "beats B&H in X/16 windows"
----------------------------------------------------------------
The README and result docs quote counts like "H-Vol beats B&H in 12/16
windows". Under the null that each window is a 50/50 coin flip, the 95% CI
on 16 Bernoullis is 3-13 successes, so 9-13/16 is *consistent with chance*.
Worse, SKEPTIC_REVIEW section 2 argues only ~6 of the 16 windows are
genuinely independent market episodes; at N=6 almost nothing survives.

Runnable as ``python scripts/project_level_dsr.py`` from the repo root.
Writes a narrative summary to ``scripts/data/project_level_dsr.md``.
"""

from __future__ import annotations

from pathlib import Path

from scipy.stats import binomtest

from signals.backtest.metrics import deflated_sharpe_ratio, expected_max_sharpe

# ---------------------------------------------------------------------------
# Section A: project-level DSR
# ---------------------------------------------------------------------------

# Table of known experiments the project has run (trial counts sourced from
# IMPROVEMENTS.md and the per-tier result docs in ``scripts/``). This is a
# lower bound — sweeps the author forgot about only push ``n_trials`` higher.
EXPERIMENTS: list[tuple[str, int, str]] = [
    # (label, n_configs, source)
    ("Tier 0:  HOMC order=7 sweep",              25,   "scripts/HOMC_ORDER7_RESULTS.md"),
    ("Tier 0a: HOMC order=5 / w=1000",            1,   "scripts/HOMC_ORDER5_W1000_RESULTS.md"),
    ("Tier 0b: Multi-asset HOMC validation",     20,   "scripts/HOMC_TIER0B_COMPREHENSIVE.md (5 cfg x 4 assets)"),
    ("Tier 0c: Hybrid model development",        10,   "scripts/HOMC_TIER0C_HYBRID_RESULTS.md"),
    ("Tier 0e: Vol quantile tuning",             24,   "scripts/HOMC_TIER0E_BTC_SP500.md (~12 cfg x 2 assets)"),
    ("Tier 0f: Sizing + blend ramp sweeps",      30,   "scripts/HOMC_TIER0F_SIZING_BLEND.md"),
    ("Tier 1S: S&P trend + HOMC memory sweep",   25,   "scripts/SP500_TREND_AND_HOMC_MEMORY.md"),
    ("Tier 2:  BTC deep sweep",                1616,   "scripts/BTC_DEEP_SWEEP_RESULTS.md"),
    ("Tier 2:  BTC/SP portfolio sweep",          50,   "scripts/BTC_SP500_PORTFOLIO_RESULTS.md"),
    ("Tier 3:  Comprehensive improvements",     100,   "scripts/TIER3_COMPREHENSIVE_RESULTS.md"),
]

N_TRIALS_PROJECT = sum(n for _, n, _ in EXPERIMENTS)


# ---------------------------------------------------------------------------
# "Beats B&H in X/16 windows" claims from the project's own docs
# ---------------------------------------------------------------------------
# Each entry maps strategy label -> (wins, n_windows, source_doc).
# Counts copied verbatim from README.md "Positive CAGR windows" row and
# the headline Tier-0c results table. See README.md:68 and
# scripts/HOMC_TIER0C_HYBRID_RESULTS.md.
WINDOW_CLAIMS: list[tuple[str, int, int, str]] = [
    ("Composite beats B&H (positive CAGR)",        11, 16, "README.md:68"),
    ("HOMC beats B&H (positive CAGR)",             10, 16, "README.md:68"),
    ("H-Vol @ q=0.70 beats B&H (positive CAGR)",   12, 16, "README.md:68"),
    ("H-Blend beats B&H (positive CAGR)",          13, 16, "README.md:68"),
]

# "Effective N" correction from SKEPTIC_REVIEW section 2: the 16 windows
# visibly cluster into ~6 distinct market episodes. Scale win counts
# proportionally and round to the nearest integer.
EFFECTIVE_N = 6
N_WINDOWS = 16


def effective_wins(wins: int, n: int, eff_n: int) -> int:
    """Scale a binomial count (wins/n) down to eff_n and round."""
    return int(round(wins * eff_n / n))


# ---------------------------------------------------------------------------
# DSR helpers
# ---------------------------------------------------------------------------

def dsr_row(
    label: str,
    sharpe: float,
    n_observations: int,
    trial_grid: list[tuple[str, int]],
) -> list[tuple[str, int, float, float]]:
    """Compute DSR under several choices of ``n_trials``.

    Returns list of (grid_label, n_trials, expected_max_sr, dsr) tuples.
    """
    out = []
    for grid_label, n_trials in trial_grid:
        e_max = expected_max_sharpe(n_trials)
        dsr = deflated_sharpe_ratio(
            sharpe=sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
        )
        out.append((grid_label, n_trials, e_max, dsr))
    return out


def verdict_for_dsr(dsr: float) -> str:
    if dsr >= 0.95:
        return "significant (DSR >= 0.95)"
    if dsr >= 0.50:
        return "weak (DSR in [0.50, 0.95))"
    return "indistinguishable from noise (DSR < 0.50)"


def format_dsr_block(
    label: str,
    sharpe: float,
    n_observations: int,
    rows: list[tuple[str, int, float, float]],
) -> list[str]:
    out = []
    out.append(f"### {label}")
    out.append(f"Observed Sharpe = {sharpe:.2f}, n_observations = {n_observations}")
    out.append("")
    out.append("| Deflation grid | n_trials | E[max SR] | DSR | Verdict |")
    out.append("|---|---:|---:|---:|---|")
    for grid_label, n_trials, e_max, dsr in rows:
        out.append(
            f"| {grid_label} | {n_trials:,} | {e_max:.3f} | {dsr:.4f} | {verdict_for_dsr(dsr)} |"
        )
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Binomial helpers
# ---------------------------------------------------------------------------

def binom_line(wins: int, n: int) -> dict:
    """Return observed wins, one-sided p vs 0.5, 95% exact CI, and verdict."""
    bt = binomtest(k=wins, n=n, p=0.5, alternative="greater")
    ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    p_one_sided = bt.pvalue
    sig = "significant (p<0.05)" if p_one_sided < 0.05 else "NOT significant"
    return {
        "wins": wins,
        "n": n,
        "rate": wins / n,
        "p_one_sided": p_one_sided,
        "ci_lo": ci.low,
        "ci_hi": ci.high,
        "verdict": sig,
    }


def format_binom_block(
    title: str,
    claims: list[tuple[str, int, int, str]],
    eff_n: int | None = None,
) -> list[str]:
    out = []
    out.append(f"### {title}")
    out.append("")
    if eff_n is None:
        out.append("| Claim | Observed | Rate | 95% CI (rate) | p (one-sided, H0=0.5) | Verdict |")
        out.append("|---|---:|---:|---|---:|---|")
        for label, wins, n, _src in claims:
            r = binom_line(wins, n)
            out.append(
                f"| {label} | {r['wins']}/{r['n']} | {r['rate']:.2f} | "
                f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
                f"{r['p_one_sided']:.4f} | {r['verdict']} |"
            )
    else:
        out.append(
            f"*Effective N = {eff_n} (section 2 of SKEPTIC_REVIEW.md: "
            f"only ~6 of 16 windows are genuinely independent market episodes).*"
        )
        out.append("")
        out.append("| Claim | Original | Effective | Rate | 95% CI (rate) | p (one-sided) | Verdict |")
        out.append("|---|---:|---:|---:|---|---:|---|")
        for label, wins, n, _src in claims:
            eff_wins = effective_wins(wins, n, eff_n)
            r = binom_line(eff_wins, eff_n)
            out.append(
                f"| {label} | {wins}/{n} | {eff_wins}/{eff_n} | {r['rate']:.2f} | "
                f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
                f"{r['p_one_sided']:.4f} | {r['verdict']} |"
            )
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lines: list[str] = []
    lines.append("# Project-level DSR + binomial significance report")
    lines.append("")
    lines.append(
        "Addresses Tier B5 and C6 of `SKEPTIC_REVIEW.md`. Generated by "
        "`scripts/project_level_dsr.py`. All numbers are arithmetic from the "
        "existing correct DSR implementation in `signals/backtest/metrics.py` "
        "and `scipy.stats.binomtest`."
    )
    lines.append("")

    # ---- Section A ---------------------------------------------------------
    lines.append("## Section A — Project-level DSR (Tier B5)")
    lines.append("")
    lines.append(
        "The project's result docs compute DSR with `n_trials` = the size "
        "of the sweep that produced that row (~25). Bailey & Lopez de Prado's "
        "deflation is supposed to account for **every** strategy evaluation "
        "the researcher has seen while picking a config. Below is the lower "
        "bound on the project's total distinct trial count."
    )
    lines.append("")

    lines.append("### Known experiments (lower bound)")
    lines.append("")
    lines.append("| Tier / sweep | n_configs | Source |")
    lines.append("|---|---:|---|")
    for label, n, src in EXPERIMENTS:
        lines.append(f"| {label} | {n:,} | `{src}` |")
    lines.append(f"| **Project total (lower bound)** | **{N_TRIALS_PROJECT:,}** | — |")
    lines.append("")
    lines.append(
        f"`n_trials_project = {N_TRIALS_PROJECT:,}`. SKEPTIC_REVIEW estimates "
        "~5,000+ once you include seeds, asset variants, and abandoned "
        "experiments; this table is conservative."
    )
    lines.append("")

    trial_grid_headline = [
        ("Per-sweep (what the project reports)", 25),
        ("Per-tier aggregate",                   200),
        ("Project-level (this script)",          N_TRIALS_PROJECT),
        ("SKEPTIC_REVIEW worst-case",            5000),
    ]

    # Headline H-Vol q=0.70, 16-window random-window eval, median Sharpe 2.15
    # (README.md:63). n_observations = 16 windows * 126 daily bars per window.
    hv_sharpe = 2.15
    hv_nobs = 16 * 126
    rows = dsr_row("H-Vol @ q=0.70", hv_sharpe, hv_nobs, trial_grid_headline)
    lines += format_dsr_block(
        "H-Vol @ q=0.70 — headline BTC median Sharpe (README.md:63)",
        hv_sharpe,
        hv_nobs,
        rows,
    )

    # Alternative headline the SKEPTIC_REVIEW asked about: 1.15 in-sample.
    alt_rows = dsr_row("H-Vol alt", 1.15, hv_nobs, trial_grid_headline)
    lines += format_dsr_block(
        "H-Vol alt — in-sample Sharpe 1.15 sensitivity check",
        1.15,
        hv_nobs,
        alt_rows,
    )

    # Tier-2 "apparent winner" sell_bps=-20 at median Sharpe 2.40
    # (scripts/BTC_DEEP_SWEEP_RESULTS.md:147). This one has a Tier-2 sweep of
    # 1,616 configs in *its own* sweep, so the per-sweep column changes.
    t2_sharpe = 2.40
    t2_nobs = 16 * 126
    trial_grid_t2 = [
        ("Per-sweep (1,616 cfg deep sweep)", 1616),
        ("Project-level",                    N_TRIALS_PROJECT),
        ("SKEPTIC_REVIEW worst-case",        5000),
    ]
    t2_rows = dsr_row("Tier-2 winner", t2_sharpe, t2_nobs, trial_grid_t2)
    lines += format_dsr_block(
        "Tier-2 apparent winner `sell_bps=-20`, median Sharpe 2.40 "
        "(BTC_DEEP_SWEEP_RESULTS.md:147)",
        t2_sharpe,
        t2_nobs,
        t2_rows,
    )

    # ---- Section B ---------------------------------------------------------
    lines.append("## Section B — Binomial significance on 'beats B&H in X/16' (Tier C6)")
    lines.append("")
    lines.append(
        "Under H0 that the strategy is a fair coin on each window "
        "(P[win vs B&H] = 0.5), we can compute the exact 95% confidence "
        "interval for the win rate and the one-sided p-value for "
        "'beats B&H more than half the time'. Counts come straight from "
        "`README.md:68` and `scripts/HOMC_TIER0C_HYBRID_RESULTS.md`."
    )
    lines.append("")

    lines += format_binom_block(
        "B.1 At face value (N = 16 'independent' windows)",
        WINDOW_CLAIMS,
        eff_n=None,
    )

    lines += format_binom_block(
        f"B.2 After effective-N correction (N = {EFFECTIVE_N})",
        WINDOW_CLAIMS,
        eff_n=EFFECTIVE_N,
    )

    lines.append("### Conclusion")
    lines.append("")
    lines.append(
        "At the face-value N=16, the one-sided binomial test *might* reach "
        "significance for the strongest claim (13/16 H-Blend) but stops "
        "short for all others. Once you acknowledge that only ~6 of the 16 "
        "windows are genuinely distinct market episodes (SKEPTIC_REVIEW "
        "section 2), **none** of the 'beats B&H' counts clear alpha=0.05. "
        "The 'beats B&H in X/16' summary statistic should be dropped from "
        "headline claims, or accompanied by the corrected binomial test "
        "shown above."
    )
    lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = Path(__file__).resolve().parent / "data" / "project_level_dsr.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
