# Quality, testing and reproducibility

[Documentation index](README.md)

## Implemented versus proposed

The supplied code does not include automated tests, CI workflows, static typing enforcement, structured logs, input schemas, a run manifest or a model registry. This page defines a recommended quality plan; it does not report executed or passing tests.

## Existing engineering safeguards

- Module boundaries separate orchestration, data, plots and evaluation.
- A shared seed controls configured stochastic behavior.
- Stratified train/test and CV splits preserve class representation.
- Preprocessing is inside model pipelines.
- The holdout is not passed into model selection.
- Figures are closed after writing.
- Output location is anchored to the entry-point source file.

## Recommended test matrix

| Level | Target | Assertion |
| --- | --- | --- |
| Unit | `load_dataset` | Expected shape, numeric features, aligned index, labels 0/1/2 |
| Unit | `split_dataset` | 142/36 rows, disjoint indices, union equals source indices |
| Unit | `split_dataset` | Repeated calls with unchanged inputs produce identical partitions |
| Unit | `get_model_configs` | Three keys; fresh estimator instances; valid parameter paths |
| Unit | Pipelines | Scaler exists for logistic regression and k-NN, not the tree |
| Unit | Analysis | CSVs appear in a temporary directory with expected fields |
| Unit | Final evaluation | Known fake predictions yield expected metrics and error rows |
| Unit | Error export | Zero mistakes produce a readable header-only CSV |
| Integration | Full run | Expected files exist and all six comparison rows are present |
| Integration | Plot generation | PNGs are nonempty and readable |
| Regression | Winner selection | Ranking uses only tuned mean CV macro F1; exact ties are stable |
| Methodology | Selection boundary | Holdout is never passed to tuning or learning-curve functions |

Use temporary directories rather than committed `main/results/` for tests. Tests that execute the complete grid search belong in a slower integration group. A fake estimator or reduced test configuration can speed unit tests without altering the documented production experiment.

## Suggested tooling

A future `pytest` suite could live under `tests/`. Ruff can provide linting and formatting, while type annotations plus a type checker can make contracts easier to maintain. These tools are optional proposed additions, not runtime dependencies already required by the scripts. Configure imports deliberately because this project currently uses sibling script imports.

## Smoke-test procedure

1. Create or activate a clean environment.
2. Install committed dependency versions if available.
3. Run the documented entry point.
4. Verify 142/36 split and six comparison rows.
5. Check all reported metrics lie in [0, 1].
6. Confirm prediction rows equal 36 and error rows equal the number of false `correct` values.
7. Confirm confusion-matrix totals equal 36.
8. Inspect warnings and preserve them with experiment notes.

Do not hard-code a desired classifier winner or score into acceptance criteria. A functioning experiment can select a different winner after legitimate environment or configuration changes.

## Reproducibility record

Record the Git commit, Python version, package versions, OS, random seed, train/test policy, CV policy, search spaces, chosen parameters and output location. The current workflow saves chosen parameters and CV results but does not automatically capture all this metadata.

Commit source and documentation. Ignore `.venv/`, `__pycache__/` and transient caches. Whether to commit result snapshots is a repository policy decision: if committed, tie them to a known run. Do not infer license rights from the presence of a LICENSE filename; inspect its contents before redistribution.

## CI recommendation

A future CI job should install reproducible dependencies, run linting and unit tests, then optionally execute a smoke experiment and upload results as artifacts. Distinguish fast checks from slow exhaustive-search integration checks. No CI pipeline is provided by this documentation package.
