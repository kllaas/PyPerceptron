# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import matplotlib.pyplot as plt

import pandas as pd
from mlxtend.plotting import plot_decision_regions

from perceptron import Perceptron

df = pd.read_csv('/home/alexey/PycharmProjects/PyPerceptron/data05.csv', header=None)

y = df.iloc[1:78, 2].values
X = df.iloc[1:78, [0, 1]].values

p = Perceptron(epochs=100, learning_rate=0.01)

p.train(X, y)
print('Weights: %s' % p.weights)


df = pd.read_csv('/home/alexey/PycharmProjects/PyPerceptron/test_data.csv', header=None)

y = df.iloc[1:21, 2].values
X = df.iloc[1:21, [0, 1]].values

print 'Test set:'

for i in range(1, len(X)):
    print ' %i/%i) Prdeict: %r' % (i, len(X), p.predict(X[i]) == y[i])

plot_decision_regions(X, y, clf=p)
plt.title('Perceptron')
plt.xlabel('firstParam')
plt.ylabel('secondParam')
plt.show()

plt.plot(range(1, len(p.errors_) + 1), p.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Errors')
plt.show()
