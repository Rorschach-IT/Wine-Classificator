from sklearn.datasets import load_wine
import pandas as pd

# Load the Wine dataset from scikit-learn
wine = load_wine()

# Create a DataFrame containing all 13 numeric features
features = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Add a column with the numeric class label (0, 1 or 2)
features["class"] = wine.target

# Add a readable class name, e.g. class_0, class_1, class_2
features["class_name"] = [wine.target_names[label] for label in wine.target]

# Print basic dataset information
print("Number of samples:", features.shape[0])
print("Number of features:", len(wine.feature_names))
print("Classes:", list(wine.target_names))

# Print the first 10 samples in a readable table
print("\nFirst 10 samples:")
print(features.head(10).to_string(index=False))

# Print how many samples belong to each class
print("\nSamples per class:")
print(features["class_name"].value_counts().to_string())