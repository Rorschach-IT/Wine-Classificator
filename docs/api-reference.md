# Python API reference

[Documentation index](README.md)

Signatures below reflect the tutorial implementation, not runtime introspection of uploaded files. Types are descriptive: the supplied functions do not declare type annotations or enforce these contracts themselves. Functions without an explicit return statement return `None`.

## `dataset.py`

### `RANDOM_STATE = 42`

Shared integer seed used for splitting, shuffled folds, supported estimators and learning-curve shuffling. It improves repeatability within an environment, not universal bitwise reproducibility.

### `load_dataset()`

Returns `(X, y, class_names)` from `load_wine(as_frame=True)`:

- `X`: pandas DataFrame, shape `(178, 13)`, numeric features.
- `y`: pandas Series, length 178, class IDs 0, 1, 2.
- `class_names`: array of three human-readable names indexed by class ID.

No parameters, file writes or explicit exception handling.

### `split_dataset(X, y)`

Returns `(X_train, X_test, y_train, y_test)` using `test_size=0.2`, `stratify=y` and the shared seed. Expects aligned samples and targets accepted by `train_test_split`. With the supplied dataset, partition sizes are 142 and 36. Original sample indices are preserved.

## `analysis.py`

### `analyze_dataset(X, y, output_dir, prefix)`

Expects a numeric DataFrame, an aligned target Series, an existing writable Path, and a filename prefix string. Prints shape, dtypes, first five rows, feature missing counts, target missing count, class counts and `X.describe().T`.

Writes `{prefix}_statistics.csv`, `{prefix}_missing_values.csv`, `{prefix}_class_counts.csv`. Returns `None`. Reports missing values but does not impute, reject or otherwise clean them.

## `visualization.py`

### `create_visualizations(X, y, class_names, output_dir)`

Calls the following three functions in sequence; returns `None`.

### `plot_class_counts(y, class_names, output_dir)`

Writes `01_class_counts.png`. Uses integer target labels as indices into `class_names`; arbitrary non-contiguous or string labels require code changes.

### `plot_histograms(X, output_dir)`

Writes `02_histograms.png`. Calls `X.hist(bins=15, ...)` to plot all numeric features. Uses the resulting Matplotlib figure and closes it after saving.

### `plot_correlations(X, output_dir)`

Writes `03_correlations.png` from `X.corr()`. Uses a fixed color range of -1 to 1 and feature-name tick labels. Intended for the numeric Wine features.

## `models.py`

### `get_model_configs()`

Returns an insertion-ordered dictionary keyed by `logistic_regression`, `knn`, and `decision_tree`. Each value has `estimator` (an unfitted Pipeline) and `params` (a GridSearchCV-compatible dictionary). Each call constructs fresh estimator instances. See [methodology](ml-methodology.md) for exact parameter grids.

## `evaluation.py`

### `SCORING`

Maps `accuracy`, `precision_macro`, `recall_macro`, `f1_macro` to corresponding scikit-learn scoring strings.

### `compare_and_tune_models(X_train, y_train, output_dir)`

Runs default-configuration cross-validation and a multimetric grid search for every configuration. Uses the same stratified five-fold splitting scheme. Grid search refits the best candidate on all supplied training data using `refit="f1_macro"`.

Writes one grid-search CSV per family and `model_comparison.csv`. Prints progress, best parameters and the comparison. Returns `(winner_name, winner)` where the name is a string and the winner is an already-fitted Pipeline. The function does not accept or evaluate the holdout.

Ranking is by tuned mean CV macro F1 descending. The stable sort preserves configuration order for exact ties: logistic regression, k-NN, decision tree. This is an implementation tie behavior, not evidence that one tied model is superior.

### `plot_learning_curve(model, X_train, y_train, output_dir)`

Evaluates cloned versions of the supplied estimator across five training-size fractions from 0.4 to 1.0, within stratified five-fold CV. Scores macro F1. Writes `05_learning_curve.png`, closes the figure and returns `None`. Size fractions refer to each fold's available training partition, not the entire original dataset. Does not replace the fitted winner returned by grid search.

### `evaluate_on_test(model_name, model, X_test, y_test, class_names, output_dir)`

Requires an already-fitted estimator compatible with `X_test`. Predicts class IDs, calculates accuracy and macro precision/recall/F1, writes a class report, a confusion matrix, all test predictions and misclassified samples. Returns `None`.

Labels are constructed as `range(len(class_names))`. `zero_division=0` explicitly assigns zero when a precision/recall/F1 calculation is undefined. This setting is used in final evaluation, not explicitly configured in the CV scorer strings.

## `main.py`

### `main()`

Creates `main/results/`, loads/splits data, prints basic whole-dataset checks, analyzes training data, creates plots, selects a model, plots its learning curve and evaluates the holdout. Returns `None`. Invoked under `if __name__ == "__main__":`.

## Error contract

No project-specific exception classes are defined. Filesystem, pandas, Matplotlib and scikit-learn exceptions/warnings propagate or follow library defaults. In particular, some invalid candidate fits can be reported by grid-search warnings rather than stopping every candidate. Missing columns, incompatible labels, invalid CV configurations and unwritable output paths are not validated by a separate application layer.
