import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from perceptron_lab.data import generate_data, load_iris_data
from perceptron_lab.model import Perceptron


def plot_decision_boundary(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.int64], model: Perceptron
) -> None:
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    points = np.c_[xx.ravel(), yy.ravel()]
    predictions = [model.predict(point) for point in points]
    Z = np.array(predictions).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="red")

    plt.title("Граница решений перцептрона")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.show()


def plot_errors(errors: list[int]) -> None:
    plt.plot(range(1, len(errors) + 1), errors, marker="o")

    plt.title("Ошибки при обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Кол-во ошибок")
    plt.show()


def main() -> None:
    X, y = generate_data()

    model = Perceptron(n_features=2)
    model.train(X, y)

    accuracy = model.score(X, y)
    print(f"Точность (кластеры): {accuracy:.2%}")

    plot_decision_boundary(X, y, model)
    plot_errors(model.errors)

    X_iris, y_iris = load_iris_data()

    iris_model = Perceptron(n_features=2)
    iris_model.train(X_iris, y_iris)

    iris_accuracy = iris_model.score(X_iris, y_iris)
    print(f"Точность (iris): {iris_accuracy:.2%}")


if __name__ == "__main__":
    main()
