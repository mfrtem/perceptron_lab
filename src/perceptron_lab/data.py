import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris, make_blobs  # type: ignore[import-untyped]


def generate_data(
    samples: int = 100,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    X, y = make_blobs(n_samples=samples, centers=2, n_features=2, random_state=42)
    return X, y


def load_iris_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    iris = load_iris()

    X = iris.data[:100, :2]
    y = iris.target[:100]
    return X, y
