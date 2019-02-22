#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

p=3
sigma = 0.1
beta_true = [0.3, -0.1, 0.7, 0.8, 0.5, -0.4]

def u(A, X, beta):
    if type(A) == int:
        if A == 1:
            return(X.dot(beta[p:]))
        else:
            return(X.dot(beta[:p]))
    else:
        return((1 - A) * X.dot(beta[:p]) + A * X.dot(beta[p:]))

def gu(A, X, beta):
    return(np.vstack(((1 - A) * X.T, A * X.T)))

def logit(A, X, beta):
    return(1/(1 + np.exp(-u(A, X, beta))))

def LinearGenerator(n=1, X=None, A=None, generate_Y=False):
    if X is None:
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1:] = np.random.normal(0, 1, size = (n, (p - 1)))
    else:
        assert X.shape[1] == p
        n = X.shape[0]
    if not generate_Y:
        return(X)
    elif A is not None:
        # if A not 0 or 1, raise error
        Y = u(A, X, beta_true) + np.random.normal(0, sigma, n)
        return((X, A, Y))
    else:
        Y0 = u(0, X, beta_true) + np.random.normal(0, sigma, n)
        Y1 = u(1, X, beta_true) + np.random.normal(0, sigma, n)
        Y = {'0': Y0, '1': Y1}
        return((X, A, Y))

def LogisticGenerator(n=1, X=None, A=None, generate_Y=False):
    if X is None:
        X = np.random.normal(0, 1, n * (p - 1)).reshape(n, p - 1)
        X = np.hstack((np.ones(n).reshape(n, 1), X))
    else:
        X = np.asarray(X).reshape(-1, p)
        n = X.shape[0]
    if not generate_Y:
        return(X)
    elif A is not None:
        # if A not 0 or 1, raise error
        # if size of A does not match X.shape[0], raise error
        Y = np.random.binomial(1, logit(A, X, beta_true))
        return((X, A, Y))
    else:
        Y0 = np.random.binomial(1, logit(0, X, beta_true))
        Y1 = np.random.binomial(1, logit(1, X, beta_true))
        Y = {'0': Y0, '1': Y1}
        return((X, A, Y))

def MSELoss(beta, O, return_second_moment = False):
    X, A, Y = O
    n = X.shape[0]
    e = u(A, X, beta) - Y
    if not return_second_moment:
        l = np.dot(e, e)/(2 * n)
        grad = np.dot(gu(A, X, beta), e)/n
        return((l, grad))
    else:
        gu_v = gu(A, X, beta)
        grad_v = gu_v * e
        Sigma = grad_v.dot(grad_v.T)/n
        Hessian = gu_v.dot(gu_v.T)/n
        return((Sigma, Hessian))

def CELoss(beta, O):
    X, A, Y = O
    mu = logit(A, X, beta)
    n = X.shape[0]
    l = np.mean(-Y * np.log(mu) - (1 - Y) * np.log(1 - mu))
    grad = np.dot(gu(A, X, beta), mu - Y)/n
    return((l, grad))

class model:
    def __init__(self, mu, generator, loss, lr, eps):
        self.mu = mu
        self.generator = generator
        self.loss = loss
        self.lr = lr
        self.eps = eps
        
    def Initialize(self):
        self.beta_hat = np.zeros(2 * p)
        self.beta_bar = np.zeros(2 * p)
        self.data = []
        self.pi_log = [] # start from step 0
        self.beta_hat_log = [] # start from step 1
        self.loss_log = []
        self.step = 0
    
    def SGD(self, T, use_IPW = True, explore = False):
        for t in range(self.step, self.step + T):
            # observe X_t
            X = self.generator()
            if explore:
                self.pi_hat = 0.5
            else:
                # update pi_hat_t-1(X_t)
                epsilon = self.eps(t)
                d = int(self.mu(1, X, self.beta_bar) > self.mu(0, X, self.beta_bar))
                self.pi_hat = (1 - epsilon) * d + epsilon/2
            # sample A_t from Bernoulli pi_hat
            A = np.random.binomial(1, self.pi_hat)
            # observe Y_t
            O = self.generator(X = X, A = A, generate_Y = True)
            # update beta_hat_t
            learning_rate = self.lr(t + 1)
            l, grad = self.loss(self.beta_hat, O)
            if use_IPW:
                grad *= (A/self.pi_hat + (1 - A)/(1 - self.pi_hat))/2
            self.beta_hat -= learning_rate * grad
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar)/(t + 1)
            # log
            self.data.append(O)
            self.pi_log.append(self.pi_hat)
            self.beta_hat_log.append(self.beta_hat.copy())
            #beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
        # return()
    
    def PluginParameterVariance(self, use_IPW = True, return_SH = False):
        S_hat = np.zeros((2 * p, 2 * p))
        H_hat = np.zeros((2 * p, 2 * p))
        IPW = 1
        for t in range(self.step):
            O = self.data[t]
            beta = self.beta_hat_log[t] # or beta_bar?
            Sigma, Hessian = self.loss(beta, O, return_second_moment = True)
            if use_IPW:
                A = O[1]
                IPW = (A/self.pi_log[t] + (1 - A)/(1 - self.pi_log[t]))/2
            S_hat += Sigma * IPW**2
            H_hat += Hessian * IPW
        S_hat /= self.step
        H_hat /= self.step
        H_hat_inv = np.linalg.inv(H_hat)
        V_hat = np.dot(np.dot(H_hat_inv, S_hat), H_hat_inv)
        if return_SH:
            return(S_hat, H_hat)
        else:
            return(V_hat)

def lr1(t):
    return(1 * t**(-0.9))

def eps(t):
    if t < 20:
        return(1)
    else:
        return(min(1, 0.1 * np.log(t)/np.sqrt(t)))

LinearModel = model(mu = u, generator = LinearGenerator, loss = MSELoss,
                   lr = lr1, eps = eps)

def MC(model, T, M):
    beta_bar = np.zeros((M, 2 * p))
    beta_bar_se = np.zeros((M, 2 * p))
    for i in range(M):
        model.Initialize()
        model.SGD(T)
        V_hat = model.PluginParameterVariance()
        beta_bar[i, :] = model.beta_bar
        beta_bar_se[i, :] = np.sqrt(np.diag(V_hat)/T)
    avg_se = np.mean(beta_bar_se, axis = 0)
    mcsd = np.std(beta_bar, axis = 0)
    ratio = avg_se/mcsd
    return(ratio, beta_bar, beta_bar_se)

def MC1(model, T):
    np.random.seed()
    model.Initialize()
    model.SGD(T)
    V_hat = model.PluginParameterVariance()
    return(model.beta_bar, np.sqrt(np.diag(V_hat)/T))

def ParallelMC(model, T, M, num_procs):
    pool = Pool(num_procs)
    _MC1_ = partial(MC1, model = model, T = T)
    experiments = [pool.apply_async(_MC1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    beta_bar_se = [e.get()[1] for e in experiments]
    avg_se = np.mean(beta_bar_se, axis = 0)
    mcsd = np.std(beta_bar, axis = 0)
    ratio = avg_se/mcsd
    return(ratio, beta_bar, beta_bar_se)


if __name__ == '__main__':
    print(' ' * 27 + 'beta01 beta02 beta03 beta11 beta12 beta13')
    for T in [1000, 10000]:
        ratio, _, __ = ParallelMC(LinearModel, T = T, M = 200, num_procs = 8)
        print('ASE/MCSD at step {:7}:'.format(T), np.round(ratio, 4))
