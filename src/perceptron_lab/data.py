from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris

def generate_data(samples = 100):
    X, y = make_blobs(n_samples=samples, centers=2, n_features=2, random_state=42)
    return X, y

def load_iris_data():
    iris = load_iris()

    X = iris.data[:100, :2]
    y = iris.target[:100]
    return X, y