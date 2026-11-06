from perceptron_lab.model import Perceptron


def test_activate():
    model = Perceptron()

    assert model.activate(1) == 1
    assert model.activate(-1) == 0


def test_predict():
    model = Perceptron(n_features=2)
    model.w = [1.0, 1.0]
    model.b = 0.0

    assert model.predict([1, 1]) == 1
    assert model.predict([-1, -1]) == 0


def test_train():
    X = [[-1, -1], [1, 1]]
    y = [0, 1]

    model = Perceptron(n_features=2)
    model.train(X, y)

    assert model.score(X, y) == 1.0


def test_score():
    X = [[-1, -1], [1, 1]]
    y = [0, 1]

    model = Perceptron(n_features=2)
    model.w = [1.0, 1.0]
    model.b = 0.0

    assert model.score(X, y) == 1.0
