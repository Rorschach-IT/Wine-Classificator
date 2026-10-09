from sklearn.datasets import load_wine

data = load_wine()

features = data.data
targets = data.target

for row in features:
    print(row)

for target in targets:
    print(target)