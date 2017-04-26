import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=0.01, epochs=50, threshold=0.5, epsilon=0):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs
        self.epsilon = epsilon

    def train(self, X, y):
        # self.weights = [0.333206, 0.038794]

        self.weights = np.zeros(X.shape[1])
        self.errors_ = []

        for e in range(self.epochs):
            total_error = 0

            for xi, answer in zip(X, y):
                error = answer - self.predict(xi)

                # calculate new weights
                for i, item in enumerate(self.weights):
                    self.weights[i] += self.learning_rate * error * xi[i]

                # add error from current iteration to total_error
                total_error += abs(answer - self.predict(xi))

            self.errors_.append(total_error)
            print 'Epoch: %d, Error: %f' % (e, total_error)

            # break if error is less then epsilon
            if total_error <= self.epsilon:
                break

        return self

    # output = W1*X1+W2+X2
    def output(self, X):
        return np.dot(X, self.weights[:])

    def predict(self, X):
        return np.where(self.output(X) >= self.threshold, 1, 0)
