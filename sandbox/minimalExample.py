"""
    Download data
    → divide them into training and test ones
    → train the model on the training data
    → let the model predict classes for the test data
    → compare predictions with real classes
    → list accuracy
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)

print("X.shape:", X.shape)
print("y.shape:", y.shape)

print("\nFirst three samples of X:")
print(X[:3])

print("\nFirst 10 classes of y:")
print(y[:10])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nX_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nFirst 10 real classes of y_test:")
print(y_test[:10])

print("\nFirst 10 predicted classes of y_pred:")
print(y_pred[:10])

print("\nAccurancy:")
print(accuracy_score(y_test, y_pred) * 100, "%")
# ex. Based on these 13 values, the model considers this wine to be class 0