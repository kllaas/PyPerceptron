import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=0.01, epochs=50, threshold=0.5):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs

    def train(self, X, y):
        # self.weights = [0.333206, 0.038794]

        self.weights = np.zeros(X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            total_error = 0

            for xi, target in zip(X, y):
                error = target - self.predict(xi)

                for i, item in enumerate(self.weights):
                    self.weights[i] += self.learning_rate * error * xi[i]

                total_error += abs(target - self.predict(xi))

            self.errors_.append(total_error)
            print 'Epoch: %d, Error: %f' % (_, total_error)

        return self

    def output(self, X):
        return np.dot(X, self.weights[:])

    def predict(self, X):
        return np.where(self.output(X) >= self.threshold, 1, 0)
