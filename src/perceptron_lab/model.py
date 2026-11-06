"""A simple perceptron for binary classification."""

import numpy as np
import numpy.typing as npt


class Perceptron:
    def __init__(
        self, n_features: int = 2, learning_rate: float = 0.01, epochs: int = 50
    ) -> None:
        """Set up the model with its learning parameters."""
        self.numf: int = n_features
        self.learn: float = learning_rate
        self.epochs: int = epochs
        self.w: list[float] = [0.0] * n_features
        self.b: float = 0.0
        self.errors: list[int] = []

    def activate(self, z: float) -> int:
        """Decide which class a sample belongs to based on its weighted sum."""
        if z >= 0:
            return 1
        return 0

    def predict(self, x: npt.NDArray[np.float64]) -> int:
        """Predict the class of a sample using the current weights."""
        z: float = 0.0
        for i in range(self.numf):
            z += self.w[i] * x[i]
        z += self.b

        return self.activate(z)

    def train(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> None:
        """Teach the perceptron to classify samples by updating its weights."""
        self.errors = []

        for epoch in range(self.epochs):
            errors = 0
            for x, target in zip(X, y):
                prediction = self.predict(x)
                error = target - prediction
                if error != 0:
                    errors += 1
                    upd = self.learn * error
                    for i in range(self.numf):
                        self.w[i] += upd * x[i]
                    self.b += upd

            self.errors.append(errors)
            if errors == 0:
                break

    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> float:
        """Check how accurately the model classifies the given samples."""
        correct = 0
        for x, target in zip(X, y):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
