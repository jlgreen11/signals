# Contributing

Thanks for your interest in this project!

## tl;dr

**This is a personal research project and is not currently accepting code
contributions.** The investigation history in `scripts/` reflects a specific
research trajectory, and external PRs would be hard to integrate without
disrupting it.

However, **bug reports are welcome** if you find something concretely wrong:

- A failing test on `main`
- A regression in a result I previously claimed
- A lookahead leak that the regression tests don't catch
- A broken build or CI failure
- A dependency issue

Please open a regular GitHub issue with:

1. What you observed
2. What you expected
3. Steps to reproduce
4. Your environment (Python version, OS, relevant package versions)

For **security vulnerabilities**, see `SECURITY.md`.

## What counts as "concretely wrong"

I'm happy to investigate issues where:

- The code does something other than what the docs, comments, or commit
  messages say it does
- A result in a pinned `scripts/*.md` file can't be reproduced by running
  the corresponding script
- A test is flaky or order-dependent
- A claim about performance doesn't survive re-running on fresh data

I'm not currently taking issues for:

- "Please add feature X" — see `IMPROVEMENTS.md` for where this project
  is going. If your idea isn't on the roadmap, it probably isn't a fit
  for this repo right now.
- Model performance questions — the investigation docs in `scripts/`
  explain exactly what has and hasn't been tested and why. If the
  question is "have you tried X?" and X isn't there, the answer is
  "no and it's probably not a priority."
- General crypto/trading advice — this repo is research infrastructure,
  not a trading service.

## If you fork and build something cool

You're welcome to fork and modify under the MIT License (see `LICENSE`).
I'd love to hear about it — open an issue with a link — but you're not
obligated to share.

## Code style (for fork maintainers)

If you fork and want to match the house style:

- `ruff check signals tests` must be clean (enforced in CI)
- `pytest` must pass 100% (enforced in CI)
- New strategy code MUST have a lookahead regression test in
  `tests/test_lookahead.py` — the engine's timing discipline is
  load-bearing for the project's validity
- New result claims MUST include a reproducible script in `scripts/`
  and a pinned results doc (`scripts/*_RESULTS.md`)
- Defaults MUST be justified by a sweep result or a documented research
  decision, not pulled out of thin air
