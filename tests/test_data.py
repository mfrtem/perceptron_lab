from perceptron_lab.data import generate_data, load_iris_data

def test_generate_data():
    X, y = generate_data(samples=100)

    assert X.shape == (100, 2)
    assert y.shape == (100,)

def test_load_iris_data():
    X, y = load_iris_data()

    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(y) == {0, 1}