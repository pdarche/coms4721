from __future__ import division
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

ratings_train = pd.read_csv('./homework4/data/ratings_train.csv', header=None)
ratings_test = pd.read_csv('./homework4/data/ratings_test.csv', header=None)
ratings_fake = pd.read_csv('./homework4/data/ratings_fake.csv', header=None)

ratings_train.columns = ['user', 'movie', 'rating']
ratings_test.columns = ['user', 'movie', 'rating']
ratings_fake.columns = ['user', 'movie', 'rating']

A = ratings_fake.pivot(index='user', columns='movie').as_matrix()
# A = ratings_train.pivot(index='user', columns='movie').as_matrix()

def optimize_U(U, V, B, C, mu, A):
    Uis = []
    for i in range(m):
        right = 0,
        left = 0
        for j in range(n):
            right += (B[i] + C[j] + mu - A[i,j]) * V[j]
            left += np.outer(V[j], V[j].T)
        Ui = -np.dot(np.linalg.inv(left), right)
        Uis.append(Ui)
    return np.array(Uis)


def optimize_V(U, V, B, C, mu, A):
    Vjs = []
    for j in range(n):
        right = 0,
        left = 0
        for i in range(m):
            right += (B[i] + C[j] + mu - A[i, j]) * U[i]
            left += np.outer(U[i], U[i].T)
        Vj = -np.dot(np.linalg.inv(left), right)
        Vjs.append(Vj)
    return np.array(Vjs)


def optimize_B(U, V, B, C, mu, A):
    Bis = []
    for i in range(m):
        b = 0
        for j in range(n):
            b += -(np.dot(U[i], V[j]) + C[j] + mu - A[i,j])
        Bis.append(b / n)
    return np.array(Bis)


def optimize_B(U, V, B, C, mu, A):
    Bis = []
    for i in range(m):
        b = 0
        for j in range(n):
            b += -(np.dot(U[i], V[j]) + C[j] + mu - A[i,j])
        Bis.append(b / n)
    return np.array(Bis)


def optimize_C(U, V, B, C, mu, A):
    Cjs = []
    for j in range(n):
        c = 0
        for i in range(m):
            c += -(np.dot(U[i], V[j]) + B[i] + mu - A[i,j])
        Cjs.append(c / m)
    return np.array(Cjs)


def update_params(B, U, C, V):
    B = optimize_B(U, V, B, C, mu, A)
    U = optimize_U(U, V, B, C, mu, A)
    C = optimize_C(U, V, B, C, mu, A)
    V = optimize_V(U, V, B, C, mu, A)
    return (B, U, C, V)


def log_likelihood(B, U, C, V, mu, A):
    lls = 0
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            lls += (np.dot(U[i], V[j]) + B[i] + C[j] + mu - A[i,j])**2
    return -(lls.sum()) / 2

v = []
for t in range(20):
    v.append(log_likelihood(B, U, C, V, mu, A))
    print v[t]
    B, U, C, V = update_params(B, U, C, V)
pd.Series(v).plot()