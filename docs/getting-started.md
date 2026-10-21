# Getting started

[Documentation index](README.md)

## Prerequisites

Use the existing virtual environment and a Python interpreter supported by the installed library versions. The tutorial does not specify or validate a minimum Python version. Runtime dependencies are scikit-learn, pandas, matplotlib and NumPy; NumPy is imported directly by the learning-curve implementation and is also a dependency of scikit-learn.

Run commands from the repository root unless stated otherwise.

## Repository layout

```text
Wine-Classificator/
├── .venv/
├── docs/
├── main/
│   ├── main.py
│   ├── dataset.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── models.py
│   ├── evaluation.py
│   └── results/                 # created at runtime
├── sandbox/
├── screenshots/
├── README.md
└── requirements.txt            # create if absent
```

`docs/`, `sandbox/` and `screenshots/` are not read by the application. Python cache directories are generated and should not be manually maintained.

## Install dependencies on Windows

Use the environment interpreter explicitly to avoid activation and interpreter mismatches:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn pandas matplotlib numpy
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip show scikit-learn pandas matplotlib numpy
```

After confirming a working environment, capture its package versions:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

For an existing committed requirements file, install that file instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A freeze file captures installed dependencies; it does not guarantee identical results across operating systems and numerical library implementations.

## Run

```powershell
.\.venv\Scripts\python.exe main\main.py
```

If the environment is already activated:

```powershell
python main/main.py
```

Alternatively, enter `main/` and run `python main.py`. The output directory is derived from `Path(__file__).resolve().parent`, so both forms write to `main/results/`.

Do not assume `python -m main.main` works: the supplied modules use sibling absolute imports such as `from dataset import ...` rather than package-relative imports.

## Expected behavior

1. Print dataset shape and split sizes: `(178, 13)`, `(142, 13)`, `(36, 13)`.
2. Print full-dataset missing counts and class counts.
3. Analyze training data and write three exploratory figures.
4. Evaluate default configurations and search each model's hyperparameters.
5. Select the tuned model with the largest mean cross-validation macro F1.
6. Generate its learning curve.
7. Evaluate that model on the held-out test set and write diagnostic files.

Execution uses `n_jobs=-1` for model-selection work and can consume multiple CPU cores. No fixed completion time is promised. Output files with matching names are overwritten; stale files are not automatically removed.

## First-run checks

Open `main/results/model_comparison.csv`, `classification_report.txt`, and `04_confusion_matrix.png`. Confirm that all three model names appear in the comparison, each with `baseline` and `tuned` rows. Do not treat any score in an example or screenshot as an expected exact output.

The Wine dataset is bundled with scikit-learn; application execution does not require downloading a dataset. Dependency installation can require network access.
