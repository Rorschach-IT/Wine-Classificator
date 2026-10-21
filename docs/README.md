# Wine Classifier documentation

English documentation for the educational Wine-Classificator repository.

## Scope and documentation baseline

This documentation describes the six Python modules supplied in the project tutorial and assumed unchanged by the project owner: `main/main.py`, `dataset.py`, `analysis.py`, `visualization.py`, `models.py`, and `evaluation.py`. The actual local source files were not uploaded or inspected. The `evaluation.py` baseline includes all imports and functions added across the tutorial steps. No experiment was executed as part of writing these documents; scores and winning parameters are intentionally not invented.

The repository layout is taken from the supplied screenshot. Runtime paths and commands below account for the application living in `main/`, not in the repository root.

## Start here

| Need | Document |
| --- | --- |
| Install and run | [Getting started](getting-started.md) |
| Understand modules and execution | [Architecture](architecture.md) |
| Look up functions and contracts | [API reference](api-reference.md) |
| Understand data, validation and model selection | [ML methodology](ml-methodology.md) |
| Interpret files and metrics | [Output reference](outputs.md) |
| Change settings safely | [How-to guides](how-to.md) |
| Diagnose problems | [Troubleshooting](troubleshooting.md) |
| Plan tests and reproducibility checks | [Quality and testing](quality-and-testing.md) |
| Understand trade-offs and future work | [Decisions and limitations](decisions-and-limitations.md) |
| Find external technical references | [Sources](sources.md) |

## Project at a glance

- Task: supervised multiclass classification of Wine dataset samples.
- Dataset: 178 samples, 13 numerical features, three labels.
- Holdout: 142 training samples and 36 test samples.
- Estimators: logistic regression, k-nearest neighbors, decision tree.
- Model selection: stratified five-fold cross-validation and grid search on training data.
- Primary selection metric: macro F1.
- Outputs: CSV tables, PNG figures and a text classification report in `main/results/`.
- Interface: a local script; there is no implemented service, CLI argument parser or web interface.

## Documentation conventions

Code identifiers and CSV column names are kept verbatim. Console messages and plot titles in the existing Python code remain Polish; English documentation does not change their language. Recommendations are explicitly identified and do not imply that tests, CI, persistence or additional validation already exist.

The information structure follows the separation of tutorials, task guides, reference and explanation described by Diataxis; it is not a claim of compliance with a formal engineering certification.

## Maintenance

Update these documents whenever signatures, output names, split settings, metric selection or search spaces change. Keep generated measurements separate from the stable documentation. Record actual experiment results and dependency versions in the repository when submitting the assignment.
