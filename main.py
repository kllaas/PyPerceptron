# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from numpy.matlib import rand

from perceptron import Perceptron


df = pd.read_csv('D:\Мои документы\Алёша\Python Projects\PyPerceptron\data05.csv', header=None)

# setosa and versicolor
y = df.iloc[1:78, 2].values

# sepal length and petal length
X = df.iloc[1:78, [0, 1]].values

p = Perceptron(epochs=100, learning_rate=0.01)

p.train(X, y)
print('Weights: %s' % p.weights)

plot_decision_regions(X, y, clf=p)
plt.title('Perceptron')
plt.xlabel('firstParam')
plt.ylabel('secondParam')
plt.show()

plt.plot(range(1, len(p.errors_) + 1), p.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()
