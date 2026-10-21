# Architecture

[Documentation index](README.md)

## Design

The project is a local batch experiment with function-oriented modules. `main.py` orchestrates execution; modules separate data access, exploration, model definitions and evaluation. There is no database, remote API, authentication layer or persistent inference service.

```text
main.py
 ├── dataset.py             load + holdout split
 ├── analysis.py            tabular training-data summaries
 ├── visualization.py       exploratory figures
 └── evaluation.py
      ├── models.py         estimator pipelines + search spaces
      └── dataset.py        shared RANDOM_STATE
```

The application passes pandas objects between modules. It does not reload CSV outputs to continue training.

## Data flow and boundaries

`load_dataset()` returns feature data, targets and target names. `split_dataset()` creates the holdout before any scaler fitting. Only `X_train` and `y_train` enter cross-validation, hyperparameter search and learning-curve computation. `X_test` and `y_test` enter final evaluation after selection.

The full dataset is used for basic shape, missing-value and class-count inspection. Detailed descriptive analysis and exploratory figures use the training partition. This is not a claim that the test set is entirely unseen in every sense; it is excluded from fitting, tuning and model selection.

## Pipeline boundaries

Logistic regression and k-NN pipelines contain a `StandardScaler` followed by the estimator. The decision-tree pipeline contains only its estimator. Passing whole pipelines into cross-validation ensures preprocessing is refitted within each fold rather than learned from validation samples.

## Ownership of side effects

| Component | Effects |
| --- | --- |
| `dataset.py` | Returns in-memory objects; no file writes |
| `analysis.py` | Prints summaries; writes CSVs |
| `visualization.py` | Writes and closes PNG figures |
| `models.py` | Constructs unfitted estimators and parameter dictionaries |
| `evaluation.py` | Fits estimators, prints scores, writes CSV/TXT/PNG files |
| `main.py` | Creates output directory and sequences calls |

## Paths and process behavior

`main()` creates the directory adjacent to its source file, not relative to the current working directory. Helper functions expect a writable, existing `pathlib.Path` output directory and do not create it themselves. Parallel work is delegated to scikit-learn/joblib through `n_jobs=-1`. The entry-point guard must remain in place.

There is no transactional output writing. An exception can leave a partially updated results directory. There is no custom exception handling or configured logging; failures normally surface as Python exceptions and console output.
