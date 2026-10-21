from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dataset import RANDOM_STATE

# Returns dictionary, here is a little database
# In pipeline the classificator is called: model
# model__max_depth means: step "model" => parameter "max_depth"
def get_model_configs():
    return {
        "logistic_regression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            "params": {
                "model__C": [0.01, 0.1, 1, 10, 100],
            },
        },
        "knn": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "decision_tree": {
            "estimator": Pipeline([
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            "params": {
                "model__max_depth": [2, 3, 4, 5, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
        },
    }