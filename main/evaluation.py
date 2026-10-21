import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from dataset import RANDOM_STATE
from models import get_model_configs

# For adding curve of learning
from sklearn.model_selection import learning_curve


SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

# In the cross_validate results the key: "test_f1_macro" means the result on the validation fragment of a given fold, not on our set aside X_test.
def compare_and_tune_models(X_train, y_train, output_dir):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    searches = {}

    for name, config in get_model_configs().items():
        print(f"\n=== Model: {name} ===")

        baseline = cross_validate(
            estimator=config["estimator"],
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=-1,
        )

        baseline_row = {
            "model": name,
            "stage": "baseline",
            "train_f1_macro": baseline["train_f1_macro"].mean(),
            "best_params": "{}",
        }

        for metric in SCORING:
            baseline_row[f"cv_{metric}"] = (
                baseline[f"test_{metric}"].mean()
            )
            baseline_row[f"cv_{metric}_std"] = (
                baseline[f"test_{metric}"].std()
            )

        rows.append(baseline_row)

        search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            scoring=SCORING,
            refit="f1_macro",
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        searches[name] = search

        index = search.best_index_

        tuned_row = {
            "model": name,
            "stage": "tuned",
            "train_f1_macro": (
                search.cv_results_["mean_train_f1_macro"][index]
            ),
            "best_params": str(search.best_params_),
        }

        for metric in SCORING:
            tuned_row[f"cv_{metric}"] = (
                search.cv_results_[f"mean_test_{metric}"][index]
            )
            tuned_row[f"cv_{metric}_std"] = (
                search.cv_results_[f"std_test_{metric}"][index]
            )

        rows.append(tuned_row)

        pd.DataFrame(search.cv_results_).to_csv(
            output_dir / f"{name}_grid_search.csv",
            index=False,
        )

        print("Best parameters", search.best_params_)
        print("Average CV F1 macro:", round(search.best_score_, 4))

    results = pd.DataFrame(rows)

    results["train_cv_gap"] = (
        results["train_f1_macro"] - results["cv_f1_macro"]
    )

    results.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    print("\n=== Models comparison ===")
    print(results.to_string(index=False))

    tuned_results = results[
        results["stage"] == "tuned"
    ].sort_values(
        "cv_f1_macro",
        ascending=False,
        kind="stable",
    )

    winner_name = tuned_results.iloc[0]["model"]
    winner = searches[winner_name].best_estimator_

    print("\nWybrany model:", winner_name)

    return winner_name, winner
   
"""
    Classification_report shows precision, recall and F1 separately for each class, so it allows you to check whether a good overall result is not masking a weakness in one class.
"""
def evaluate_on_test(
    model_name,
    model,
    X_test,
    y_test,
    class_names,
    output_dir,
):
    predictions = model.predict(X_test)

    labels = list(range(len(class_names)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    metrics = pd.DataFrame([{
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }])

    print("\n=== Final Test Results ===")
    print(metrics.to_string(index=False))

    metrics.to_csv(
        output_dir / "final_test_metrics.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {model_name}")

    fig.tight_layout()
    fig.savefig(
        output_dir / "04_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    test_results = X_test.copy()
    test_results["actual"] = y_test
    test_results["predicted"] = predictions
    test_results["correct"] = (
        test_results["actual"] == test_results["predicted"]
    )

    test_results.to_csv(
        output_dir / "test_predictions.csv",
        index_label="sample_index",
    )

    errors = test_results.loc[~test_results["correct"]]

    errors.to_csv(
        output_dir / "misclassified_samples.csv",
        index_label="sample_index",
    )

    print("\nNumber of misclassified samples:", len(errors))

    if not errors.empty:
        print(errors[["actual", "predicted"]].to_string())
        
# This function below shows training and validation results for different sizes of training data.
def plot_learning_curve(model, X_train, y_train, output_dir):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.4, 1.0, 5),
        cv=cv,
        scoring="f1_macro",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        sizes,
        train_mean,
        marker="o",
        label="Training",
    )
    ax.fill_between(
        sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    ax.plot(
        sizes,
        validation_mean,
        marker="o",
        label="Validation",
    )
    ax.fill_between(
        sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )

    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "05_learning_curve.png",
        dpi=150,
    )
    plt.close(fig)