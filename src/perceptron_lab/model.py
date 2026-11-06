class Perceptron:
    def __init__(self, n_features=2, learning_rate=0.01, epochs=50):
        self.numf = n_features
        self.learn = learning_rate
        self.epochs = epochs
        self.w = [0.0] * n_features
        self.b = 0.0
        self.errors = []

    def activate(self, z):

        if z >= 0:
            return 1
        return 0

    def predict(self, x):

        z = 0
        for i in range(self.numf):
            z += self.w[i] * x[i]
        z += self.b

        return self.activate(z)

    def train(self, X, y):

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

    def score(self, X, y):

        correct = 0
        for x, target in zip(X, y):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
