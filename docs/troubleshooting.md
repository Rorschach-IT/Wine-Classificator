# Troubleshooting

[Documentation index](README.md)

## `ModuleNotFoundError: sklearn`, pandas or matplotlib

Check that installation and execution use the same interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip show scikit-learn
.\.venv\Scripts\python.exe main\main.py
```

The install package is named `scikit-learn`, while the Python import namespace is `sklearn`.

## `ModuleNotFoundError: dataset` or another sibling module

Run the script as documented: `python main/main.py` from the repository root or `python main.py` inside `main/`. The tutorial uses script-style sibling imports. It is not currently structured as an installable package with package-relative imports.

## IDE shows unresolved imports but terminal execution works

Verify the IDE project interpreter points to this repository's `.venv`. Compare it to the interpreter used by the terminal. This is an environment check, not a reason to rename imports.

## Import error for a missing evaluation function

The tutorial builds `evaluation.py` incrementally. Confirm it contains `compare_and_tune_models`, `plot_learning_curve` and `evaluate_on_test`, together with the imports introduced in each step. The final `main.py` expects all three.

## Output appears in an unexpected directory

The actual code writes next to `main.py`, into `main/results/`. It does not use root `results/`. Recheck which copy of the script was executed and whether your local code still uses the documented Path expression.

## No plot window opens

Expected behavior: plots are saved to PNG and closed. Open the files in `main/results/`. For a headless runtime, a noninteractive Matplotlib backend such as Agg can be selected if backend problems occur.

## CPU usage is high

Model search and learning curves use `n_jobs=-1`. Reduce worker counts as described in the how-to guide. Repeated fits are expected; a high CPU load is not itself evidence of a loop bug.

## Permission error or partially updated results

Ensure the output directory is writable and files are not locked by another application. Close spreadsheets viewing output CSVs when necessary. The implementation does not roll back files after failure. Preserve old results or use a clean run directory before retrying.

## Undefined metric warnings

A model can predict no samples for a class, making some metrics undefined. Final evaluation explicitly uses `zero_division=0`. Cross-validation uses named scorers with their library defaults, so warnings can still appear there. Inspect classwise performance and predictions rather than suppressing warnings without understanding them.

## Candidate fits fail after editing a grid

Check valid estimator parameter names, numeric ranges, neighbor counts relative to fold-training sizes, and class support for the chosen splitter. Inspect FitFailedWarning details. The supplied code does not add `error_score="raise"`; enabling it temporarily can help development but changes failure behavior.

## Results differ between environments

Check the seed, source revision, dependency versions and library defaults. Exact score reproduction is not guaranteed across arbitrary versions. Compare unrounded CSV numbers rather than only the four-decimal report.

## A score is 1.0 or the error CSV has no rows

This can occur and is not proof of a bug or universal accuracy. Confirm the holdout is excluded from model selection. Remember it contains only 36 samples. Do not contaminate the test by repeatedly changing the experiment based on its outcome.
