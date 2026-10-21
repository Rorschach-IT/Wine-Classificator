from pathlib import Path

from analysis import analyze_dataset
from dataset import load_dataset, split_dataset
from evaluation import (
    compare_and_tune_models,
    evaluate_on_test,
    plot_learning_curve,
)
from visualization import create_visualizations


def main():
    # Set up the results output directory
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and split into training and test sets
    X, y, class_names = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("Whole dataset:", X.shape)
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

    # Check the structure and missing values of the entire dataset
    print("\nMissing values in the whole dataset:")
    print(X.isna().sum().to_string())
    print("\nClass counts in the whole dataset:")
    print(y.value_counts().sort_index().to_string())

    # Detailed exploratory data analysis on the training set only
    analyze_dataset(
        X_train,
        y_train,
        output_dir,
        prefix="train",
    )

    # Generate and save exploratory data visualizations for training data
    create_visualizations(
        X_train,
        y_train,
        class_names,
        output_dir,
    )

    # Model comparison, hyperparameter tuning, and selection via cross-validation
    winner_name, winner = compare_and_tune_models(
        X_train,
        y_train,
        output_dir,
    )

    # Plot learning curve for the best performing model
    plot_learning_curve(
        winner,
        X_train,
        y_train,
        output_dir,
    )

    # Final evaluation of the selected model on the test set
    evaluate_on_test(
        winner_name,
        winner,
        X_test,
        y_test,
        class_names,
        output_dir,
    )


if __name__ == "__main__":
    main()