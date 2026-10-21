def analyze_dataset(X, y, output_dir, prefix):
    print(f"\n=== Analysis: {prefix} ===")

    print("\nNumber of samples and features:")
    print(X.shape)

    print("\nData types:")
    print(X.dtypes.to_string())

    print("\nFirst five samples:")
    print(X.head().to_string())

    missing = X.isna().sum()

    print("\nMissing values in features:")
    print(missing.to_string())

    print("\nMissing labels:")
    print(y.isna().sum())

    class_counts = y.value_counts().sort_index()

    print("\nClass counts:")
    print(class_counts.to_string())

    statistics = X.describe().T

    print("\nStatistics:")
    print(statistics.to_string())

    statistics.to_csv(output_dir / f"{prefix}_statistics.csv")
    missing.to_csv(
        output_dir / f"{prefix}_missing_values.csv",
        header=["missing_count"],
    )
    class_counts.to_csv(
        output_dir / f"{prefix}_class_counts.csv",
        header=["count"],
    )