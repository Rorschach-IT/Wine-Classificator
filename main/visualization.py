# This file creates charts with data visualisation
import matplotlib.pyplot as plt


def create_visualizations(X, y, class_names, output_dir):
    plot_class_counts(y, class_names, output_dir)
    plot_histograms(X, output_dir)
    plot_correlations(X, output_dir)


def plot_class_counts(y, class_names, output_dir):
    counts = y.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        [class_names[int(label)] for label in counts.index],
        counts.values,
        color="steelblue",
    )

    ax.set_title("Class Counts — Training Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")

    fig.tight_layout()
    fig.savefig(output_dir / "01_class_counts.png", dpi=150)
    plt.close(fig)


def plot_histograms(X, output_dir):
    axes = X.hist(
        bins=15,
        figsize=(15, 11),
        color="steelblue",
        edgecolor="black",
    )

    fig = axes.ravel()[0].figure
    fig.suptitle("Feature Distributions — Training Set")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(output_dir / "02_histograms.png", dpi=150)
    plt.close(fig)


def plot_correlations(X, output_dir):
    correlations = X.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(
        correlations,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    positions = range(len(X.columns))

    ax.set_xticks(
        positions,
        labels=X.columns,
        rotation=90,
    )
    ax.set_yticks(
        positions,
        labels=X.columns,
    )

    ax.set_title("Feature Correlations — Training Set")

    fig.colorbar(image, ax=ax, label="Pearson Correlation")

    fig.tight_layout()
    fig.savefig(output_dir / "03_correlations.png", dpi=150)
    plt.close(fig)