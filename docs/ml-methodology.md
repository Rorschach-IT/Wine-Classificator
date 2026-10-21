# Machine-learning methodology

[Documentation index](README.md)

## Dataset contract

The bundled Wine dataset contains 178 samples and 13 numerical chemical features. Targets are three dataset classes, not quality grades, prices or general wine recommendations. Full-dataset class counts are 59, 71 and 48 for labels 0, 1 and 2 respectively.

Features are: `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280/od315_of_diluted_wines`, `proline`.

Keep the dataset's original feature spelling when selecting columns. Feature units are not assigned or converted by application code. For provenance and chemical details, consult the dataset description rather than inferring units from column names.

## Holdout and exploration

A stratified 80/20 split with seed 42 gives 142 training samples and 36 holdout samples. The holdout is not passed into scaler fitting, CV, search or learning curves. Full-dataset structural checks occur, while detailed summaries and plots use training data.

Missing-value detection is implemented. Imputation is not. The supplied dataset is expected to have no missing values; an altered dataset with missing values requires an explicit preprocessing policy, preferably within the pipeline.

## Estimators and search spaces

| Family | Preprocessing / estimator settings | Searched parameters |
| --- | --- | --- |
| Logistic regression | StandardScaler; max_iter=3000; random_state=42 | `model__C`: 0.01, 0.1, 1, 10, 100 |
| k-NN | StandardScaler; KNeighborsClassifier | `model__n_neighbors`: 3, 5, 7, 9, 11; `model__weights`: uniform, distance; `model__p`: 1, 2 |
| Decision tree | No scaler; random_state=42 | `model__max_depth`: 2, 3, 4, 5, None; `model__min_samples_leaf`: 1, 2, 4; `model__criterion`: gini, entropy |

Candidate counts are 5, 20 and 30 respectively: 55 total. Five-fold search entails 275 candidate/fold fits plus one refit for each family. Default-model CV adds 15 fits, and the winner's learning curve adds 25 fits. These counts describe the configured workflow; they are not performance benchmarks.

In logistic regression, smaller C means stronger regularization. In k-NN, p=1 and p=2 correspond to Manhattan and Euclidean distance under the default Minkowski metric. Tree depth and minimum leaf size constrain model complexity. Defaults not set in code remain library defaults and can change between versions.

## Baseline and search

The baseline means each family's configured estimator before grid search, not a majority-class DummyClassifier. No dummy baseline is implemented.

Both baseline CV and GridSearchCV use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. All four metrics are scored. Grid search selects and refits by macro F1. The final family winner is the tuned row with the largest mean macro F1; exact ties follow configuration order.

## Metrics

- Accuracy: correct predictions divided by all predictions.
- Precision for class c: TP / (TP + FP).
- Recall for class c: TP / (TP + FN).
- F1 for class c: 2PR / (P + R), with explicit undefined-value handling where configured.
- Macro precision, recall and F1: unweighted means of their corresponding classwise values.

Macro F1 is the mean of per-class F1 values, not generally the F1 obtained by combining macro precision and macro recall. Every class receives equal weight. `classification_report` additionally displays per-class support and weighted averages.

## Interpretation boundaries

Grid-search scores are used to select candidates, so they can be optimistic estimates of the selected configuration. Searching several families adds another selection step. The held-out score is the final independent evaluation relative to those fitting and selection decisions. Nested CV would offer a different assessment of the selection procedure but is not implemented.

A `test_*` key in cross-validation output refers to a validation fold, not the final holdout. Comparison CSV columns use `cv_*` to reduce that ambiguity.

## Overfitting diagnostics

`train_cv_gap` is mean training macro F1 minus mean validation macro F1 for a configuration. A large positive gap can indicate overfitting, but there is no universal threshold and no automatic verdict in code. CV standard deviations describe variation across these folds; they are not confidence intervals.

The learning curve plots mean scores and plus/minus one standard deviation across folds for each training size. Hyperparameters were already selected using training data, so the curve is diagnostic, not an independent unbiased assessment.

With 36 holdout samples, one prediction changes accuracy by about 2.78 percentage points. Perfect holdout accuracy does not prove perfect generalization. Do not repeatedly tune after inspecting holdout errors and continue describing the same holdout as independent.

See [external sources](sources.md) for validation and leakage guidance.
