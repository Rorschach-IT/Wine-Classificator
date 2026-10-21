# Output reference

[Documentation index](README.md)

## Location and lifecycle

All runtime artifacts are written to `main/results/`. Fixed filenames are overwritten on subsequent runs. Files from obsolete configurations are not removed. There is no run manifest, timestamped directory, automatic archive or atomic multi-file commit. Preserve a run separately before changing the experiment.

## Inventory

| File | Meaning |
| --- | --- |
| `train_statistics.csv` | `X_train.describe().T`: count, mean, std, min, quartiles and max per feature |
| `train_missing_values.csv` | Feature-indexed missing counts; data column `missing_count` |
| `train_class_counts.csv` | Label-indexed counts; data column `count` |
| `01_class_counts.png` | Training label counts |
| `02_histograms.png` | Training feature histograms, 15 bins |
| `03_correlations.png` | Training feature Pearson correlation matrix |
| `logistic_regression_grid_search.csv` | All logistic-regression candidates and CV results |
| `knn_grid_search.csv` | All k-NN candidates and CV results |
| `decision_tree_grid_search.csv` | All tree candidates and CV results |
| `model_comparison.csv` | Six rows: baseline and tuned configuration for each family |
| `05_learning_curve.png` | Selected configuration's training/validation curves |
| `final_test_metrics.csv` | One row for the selected model on the holdout |
| `classification_report.txt` | Per-class and aggregate holdout metrics |
| `04_confusion_matrix.png` | Selected model's holdout confusion matrix |
| `test_predictions.csv` | Holdout features, actual label, prediction and correctness |
| `misclassified_samples.csv` | Incorrect rows only, retaining the same columns |

Numeric prefixes in plot filenames express presentation order; the learning curve is generated before final evaluation even though its filename starts with 05.

## Comparison schema

`model_comparison.csv` contains:

- `model`: configuration key.
- `stage`: `baseline` or `tuned`.
- `train_f1_macro`: mean fold-training macro F1.
- `best_params`: string representation of selected parameters, or `{}` for baseline; this is not a JSON contract.
- `cv_accuracy`, `cv_precision_macro`, `cv_recall_macro`, `cv_f1_macro`: mean validation scores.
- Corresponding `*_std` columns: validation-score standard deviations.
- `train_cv_gap`: `train_f1_macro - cv_f1_macro`.

The CSV is in generation order, not sorted by rank. Winner selection uses a separate sorted subset of tuned rows. A tuned row reports all metrics at the candidate selected by macro F1, not each metric's independently best candidate.

## Grid-search files

These are direct DataFrame serializations of `GridSearchCV.cv_results_`. Expect parameter columns, fit/score timings, per-split scores, mean/std scores and ranks. Exact library-provided details may vary by version. `test` in their column names means a CV validation fold.

## Final metrics and predictions

`final_test_metrics.csv` has `model`, `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`. Numeric values use the 0-to-1 scale, not percentage strings.

Prediction files preserve the original sample index as `sample_index`. Additional columns are `actual`, `predicted`, `correct`. A zero-error run produces a misclassification CSV with headers and zero data rows. Sample indices identify original dataset positions, not real-world customer or bottle identifiers.

## Figures and reports

Confusion-matrix rows are actual labels; columns are predicted labels. Diagonal values are correct classifications. The text report uses four decimal places; the CSV retains numerical precision as serialized by pandas. Rounded text and CSV values need not look identical.

Figures use 150 DPI and are closed after saving. The script does not call `plt.show()`. Class counts and histogram plots describe training data, while the confusion matrix describes the holdout.

No fitted model, probabilities, tree rendering or feature-importance report is saved by the current implementation.
