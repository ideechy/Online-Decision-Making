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

def MSELoss(beta, O, gradient_only = False, return_second_moment = False):
    X, A, Y = O
    n = X.shape[0]
    e = u(A, X, beta) - Y
    if not return_second_moment:
        grad = np.dot(gu(A, X, beta), e)/n
        if gradient_only:
            return(grad)
        else:
            l = np.dot(e, e)/(2 * n)
        return((l, grad))
    else:
        gu_v = gu(A, X, beta)
        grad_v = gu_v * e
        Sigma = grad_v.dot(grad_v.T)/n
        Hessian = gu_v.dot(gu_v.T)/n
        return((Sigma, Hessian))

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
        # initialization for DoubleWeighting method
        self.beta_hat_b = None
        self.beta_bar_b = None
    
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
            l, g = self.loss(self.beta_hat, O)
            if use_IPW:
                g *= (A/self.pi_hat + (1 - A)/(1 - self.pi_hat))/2
            self.beta_hat -= learning_rate * g
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar)/(t + 1)
            # log
            self.data.append(O)
            self.pi_log.append(self.pi_hat)
            self.beta_hat_log.append(self.beta_hat.copy())
            #beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
    
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
        
    def DoubleWeighting(self, T, B=100, explore = False):
        if self.beta_hat_b is None:
            self.beta_hat_b = np.zeros((B, 2 * p))
        if self.beta_bar_b is None:
            self.beta_bar_b = np.zeros((B, 2 * p))
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
            # calculate gradient
            learning_rate = self.lr(t + 1)
            W_ip = (A/self.pi_hat + (1 - A)/(1 - self.pi_hat))/2   
            l, g = self.loss(self.beta_hat, O)
            # update beta_hat_t
            self.beta_hat -= learning_rate * W_ip * g
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar)/(t + 1)
            # Resampling
            W_b = np.random.exponential(size = (B, 1))
            G = np.array([self.loss(b, O, gradient_only = True) for b in self.beta_hat_b])
            self.beta_hat_b -= learning_rate * W_ip * W_b * G
            self.beta_bar_b = (self.beta_hat_b + t * self.beta_bar_b)/(t + 1)
            # log
            self.data.append(O)
            self.pi_log.append(self.pi_hat)
            self.beta_hat_log.append(self.beta_hat.copy())
            #beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
            
    def ResamplingParameterVariance(self):
        return(np.cov(self.beta_bar_b, rowvar = False))

def lr1(t):
    return(1 * t**(-0.9))

def eps(t):
    if t < 20:
        return(1)
    else:
        return(min(1, 0.1 * np.log(t)/np.sqrt(t)))

LinearModel = model(mu = u, generator = LinearGenerator, loss = MSELoss,
                    lr = lr1, eps = eps)

def MC(model, T, B, M):
    beta_bar = np.zeros((M, 2 * p))
    beta_bar_plugin_se = np.zeros((M, 2 * p))
    beta_bar_resample_se = np.zeros((M, 2 * p))
    for i in range(M):
        model.Initialize()
        model.DoubleWeighting(T = T, B = B)
        plugin_V = model.PluginParameterVariance()
        resample_V = model.ResamplingParameterVariance()
        beta_bar[i, :] = model.beta_bar
        beta_bar_plugin_se[i, :] = np.sqrt(np.diag(plugin_V)/T)
        beta_bar_resample_se[i, :] = np.sqrt(np.diag(resample_V))
    plugin_se = np.mean(beta_bar_plugin_se, axis = 0)
    resample_se = np.mean(beta_bar_resample_se, axis = 0)
    mcsd = np.std(beta_bar, axis = 0)
    return(beta_bar, mcsd, plugin_se, resample_se)

def MC1(model, T, B):
    np.random.seed()
    model.Initialize()
    model.DoubleWeighting(T = T, B = B)
    plugin_V = model.PluginParameterVariance()
    resample_V = model.ResamplingParameterVariance()
    beta_bar_plugin_se = np.sqrt(np.diag(plugin_V)/T)
    beta_bar_resample_se = np.sqrt(np.diag(resample_V))
    return(model.beta_bar, beta_bar_plugin_se, beta_bar_resample_se)

def ParallelMC(model, T, B, M, num_procs):
    pool = Pool(num_procs)
    _MC1_ = partial(MC1, model = model, T = T, B = B)
    experiments = [pool.apply_async(_MC1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    beta_bar_plugin_se = [e.get()[1] for e in experiments]
    beta_bar_resample_se = [e.get()[2] for e in experiments]
    plugin_se = np.mean(beta_bar_plugin_se, axis = 0)
    resample_se = np.mean(beta_bar_resample_se, axis = 0)
    mcsd = np.std(beta_bar, axis = 0)
    return(beta_bar, mcsd, plugin_se, resample_se)


if __name__ == '__main__':
    print(' ' * 27 + 'beta01 beta02 beta03 beta11 beta12 beta13')
    for T in [1000, 10000, 100000]:
        _, mcsd, plugin_se, resample_se = ParallelMC(LinearModel, T = T, B = 200, M = 200, num_procs = 16)
        print('PSE/MCSD at step {:7}:'.format(T), np.round(plugin_se/mcsd, 4))
        print('RSE/MCSD at step {:7}:'.format(T), np.round(resample_se/mcsd, 4))
