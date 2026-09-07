# Contributing to FootballVision

Thanks for helping improve FootballVision. Small, focused changes with a clear
test or reproduction case are the easiest to review.

## Before opening an issue

- Search existing issues first.
- Run `./scripts/diagnose.py` and include its non-sensitive output for setup or
  capture problems.
- Describe the source type, operating system, Python version, camera angle, and
  observed behavior.
- Do not upload copyrighted match footage, model credentials, API keys, local
  database files, or personally identifying data.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
./scripts/download_weights.sh
```

Run the application with `python main.py --reload` and open
<http://localhost:8000>.

## Making a change

1. Create a branch from `main`.
2. Keep the change limited to one bug or feature.
3. Add or update tests for behavior changes.
4. Run the deterministic quality gate:

   ```bash
   ./scripts/quality_gate.sh
   ```

5. If perception, projection, team assignment, or latency changed, also run:

   ```bash
   RUN_GOLDEN=1 ./scripts/quality_gate.sh
   ```

6. Open a pull request explaining the user-facing outcome, verification, and
   any known limitations.

Generated footage, weights, databases, exports, virtual environments, and
secrets must remain untracked. The repository `.gitignore` covers the standard
locations.

## Code expectations

- Support Python 3.11 and 3.12.
- Preserve local-first behavior and keep the default path account-free.
- Fail with a useful message when capture, model, or export prerequisites are
  missing.
- Avoid overstating model accuracy; measured benchmark results belong in
  `benchmarks/`.
- Keep new network services optional and document any privacy implications.

By contributing, you agree that your contribution is licensed under the
repository's MIT License.
