# Project: gel-blot-analyzer

refer to "C:\Users\User\gel-blot-analyzer\BUILD_BRIEF.md"

## Tech stack & commands

- Language/framework: Python 3.13, Flask + SQLAlchemy (SQLite), vanilla JS/SVG frontend, PyTorch for the band-detection model.
- Install: `pip install -r requirements.txt`
- Run: `KMP_DUPLICATE_LIB_OK=TRUE python -m app.server` (serves on :5000; the env var works around a Windows/conda OpenMP double-init warning-as-error)
- Test: `KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q`
- Lint/typecheck: none configured yet.
- Train the band-detection model (offline, not part of the runtime server): `python -m app.training.train --epochs 25 --batch-size 8 --synth-per-epoch 300`. Writes `app/training/artifacts/band_detector.pt`, which `app/detection/ml_infer.py` loads lazily at inference time. Add `--resume` to continue from the existing checkpoint. A completed run writes `app/training/artifacts/training_record.json`; if that file is missing or stale, the checkpoint came from an interrupted run.
- Evaluate the model: `python -m app.training.evaluate`. Reports dice separately on synthetic and real GelGenie val/test data; use the real scores, not the mixed `val_dice` stored in the checkpoint, which is dominated by easy synthetic samples.
- Benchmark the whole detection pipeline: `python -m app.training.benchmark --name NAME --split test`. Band recall/precision/F1, merged bands, lane errors and speed on full-resolution real gels, written to `app/training/artifacts/benchmarks/`. Use this to compare model or pre-processing variants; tune on `--split val`, report `test`.
- Evaluate lane detection: `python -m app.training.evaluate_lanes`. Scores detected lanes against lanes derived from GelGenie band masks (spurious / missed / merged). The masks only label lanes that have annotated bands, so some "spurious" lanes are real lanes the annotators skipped.

## Code style

- Lean, precise, concise code. No unnecessary abstractions — three similar lines beats a premature helper.
- No emojis, in code, comments, or commit messages.
- No comments explaining WHAT code does. Only comment a non-obvious WHY (a workaround, a hidden constraint).

## Workflow contract

- For anything beyond a small fix: write or update `SPEC.md` via `/spec` before implementation starts. Don't start coding a nontrivial feature from a bare prompt.
- Before calling a feature/task done, run `/verify` against `SPEC.md`.
- List the steps you're about to take before writing code on anything nontrivial.
- Build in parts with a checkpoint between them where useful — don't silently produce one giant diff for a multi-part change.
- Never commit to `main`/`master` unless explicitly told to. Confirm the current branch before committing; if on main/master, stop and ask.
- Never force-push, rebase, or run other history-rewriting git commands without asking first.

## Project-specific rules

- Never add code that stores a reference band/lane pattern and compares a new sample against it to output a diagnosis/classification/phenotype. That is the specific mechanism claimed by US Patent 11686703 (see `BUILD_BRIEF.md`) and is out of scope for this project (see `SPEC.md` Non-goals). Plain detection + quantification only.
- The band-detection model (`app/training/`) is trained on synthetic data (`app/training/synth_data.py`) plus a real public dataset (GelGenie, Dunn Lab / University of Edinburgh, CC-BY-4.0, Zenodo record 13218469) checked out under `app/training/external_data/` (every subset that ships images: nathan_gels, matthew_gels, matthew_gels_2, quantitation_ladder_gels, stella_gels_for_finetuning; download the zips from Zenodo and unzip each into a folder of the same name). Don't delete that directory without re-running training first, or `ml_infer.py` falls back to a much weaker classical-only heuristic.
- SQLAlchemy gotcha: when replacing a relationship collection's contents (e.g. re-running detection), use `parent.children.append(...)` / `.clear()`, not a bare `session.add(child)` with the FK set manually — the latter leaves the in-memory collection stale (see git history on `app/routes.py` for the bug this caused).
- Image adjustments (crop/rotate/brightness/contrast) apply incrementally to the *currently stored* image, not replayed from the original upload — the crop rectangle the user draws is always in the current image's pixel space.
