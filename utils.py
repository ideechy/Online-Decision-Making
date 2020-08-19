#!/usr/bin/env python3
# coding: utf-8

import numpy as np


def u(A, X, beta):
    assert len(beta) % 2 == 0
    p = len(beta) // 2
    if type(A) == int:
        if A == 1:
            return X.dot(beta[p:])
        else:
            return X.dot(beta[:p])
    else:
        return (1 - A) * X.dot(beta[:p]) + A * X.dot(beta[p:])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit01(A, X, beta):
    return sigmoid(u(A, X, beta))


def gu(A, X):
    return np.vstack(((1 - A) * X.T, A * X.T))


def LinearGenerator(beta_true, sigma, n=1, X=None, A=None, generate_Y=False):
    assert len(beta_true) % 2 == 0
    p = len(beta_true) // 2
    if X is None:
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1:] = np.random.normal(0, 1, size=(n, (p - 1)))
    else:
        assert X.shape[1] == p
        n = X.shape[0]
    if not generate_Y:
        return X, A
    elif A is not None:
        # if A not 0 or 1, raise error
        Y = u(A, X, beta_true) + np.random.normal(0, sigma, n)
        return X, A, Y
    else:
        Y0 = u(0, X, beta_true) + np.random.normal(0, sigma, n)
        Y1 = u(1, X, beta_true) + np.random.normal(0, sigma, n)
        Y = {'0': Y0, '1': Y1}
        return X, A, Y


def LogisticGenerator01(beta_true, n=1, X=None, A=None, generate_Y=False):
    assert len(beta_true) % 2 == 0
    p = len(beta_true) // 2
    if X is None:
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1:] = np.random.normal(0, 1, size = (n, (p - 1)))
    else:
        assert X.shape[1] == p
    if not generate_Y:
        return X, A
    elif A is not None:
        # if A not 0 or 1, raise error
        # if size of A does not match X.shape[0], raise error
        Y = np.random.binomial(1, logit01(A, X, beta_true))
        return X, A, Y
    else:
        Y0 = np.random.binomial(1, logit01(0, X, beta_true))
        Y1 = np.random.binomial(1, logit01(1, X, beta_true))
        Y = {'0': Y0, '1': Y1}
        return X, A, Y


def MSELoss(beta, O, gradient_only=False, return_second_moment=False):
    X, A, Y = O
    n = X.shape[0]
    e = u(A, X, beta) - Y
    if not return_second_moment:
        grad = np.dot(gu(A, X), e) / n
        if gradient_only:
            return grad
        else:
            l = np.dot(e, e) / (2 * n)
        return l, grad
    else:
        gu_v = gu(A, X)
        grad_v = gu_v * e
        Sigma = grad_v.dot(grad_v.T) / n
        Hessian = gu_v.dot(gu_v.T) / n
        return Sigma, Hessian


def BCELoss(beta, O, gradient_only = False, return_second_moment = False):
    X, A, Y = O
    n = X.shape[0]
    mu = logit01(A, X, beta)
    if not return_second_moment:
        grad = np.dot(gu(A, X), mu - Y)/n
        if gradient_only:
            return grad
        else:
            l = np.mean(-Y * np.log(mu) - (1 - Y) * np.log(1 - mu))
        return l, grad
    else:
        gu_v = gu(A, X)
        grad_v = gu_v * (mu - Y)
        Sigma = grad_v.dot(grad_v.T)/n
        Hessian = np.dot(gu_v * mu * (1 - mu), gu_v.T)/n
        return Sigma, Hessian