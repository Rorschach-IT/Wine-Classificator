from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42 # we establish a random draw so that subsequent runs are comparable.

def load_dataset():
    wine = load_wine(as_frame=True)

    X = wine.data
    y = wine.target

    return X, y, wine.target_names


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2, # for end test we will get ~20% of data
        random_state=RANDOM_STATE,
        stratify=y, # we maintain approximately the proportions of classes in both parts.
    )