from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

print(X.shape)
print(X.head(20).to_string())
# print(X.describe().T)
# print(X.isna().sum())
# print(y.value_counts().sort_index())