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
    def __init__(self, mu, generator, loss, 
                 eps = None, epsilon = 0.01, 
                 alpha = 1, gamma = 2/3, lr = None):
        self.mu = mu
        self.generator = generator
        self.loss = loss
        if eps is None:
            self.eps = lambda x: epsilon
        else:
            self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        if lr is None:
            self.lr = lambda x: alpha * x**(-gamma)
        else:
            self.lr = lr
        
    def Initialize(self):
        self.beta_hat = np.zeros(2 * p)
        self.beta_bar = np.zeros(2 * p)
        self.data = []
        self.pi_log = [] # start from step 0
        #self.beta_hat_log = [self.beta_hat.copy()] # start from step 1
        self.beta_bar_log = [self.beta_bar.copy()] # start from step 1
        self.loss_log = []
        self.step = 0
        # initialization for plugin variance estimation
        self.S_hat = np.zeros((2 * p, 2 * p))
        self.H_hat = np.zeros((2 * p, 2 * p))
        self.V_hat = np.zeros((2 * p, 2 * p))
        self.se_hat = np.zeros(2 * p)
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
            # calculate IPW
            if use_IPW:
                IPW = (A/self.pi_hat + (1 - A)/(1 - self.pi_hat))/2
            else:
                IPW = 1
            # update plugin variance estimation
            S, H = self.loss(self.beta_bar, O, return_second_moment = True)
            self.S_hat += S * IPW**2
            self.H_hat += H * IPW
            # update beta_hat_t
            learning_rate = self.lr(t + 1)
            l, g = self.loss(self.beta_hat, O) 
            g *= IPW
            self.beta_hat -= learning_rate * g
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar)/(t + 1)
            # log
            self.data.append(O)
            self.pi_log.append(self.pi_hat)
            #self.beta_hat_log.append(self.beta_hat.copy())
            self.beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
        S_hat = self.S_hat/self.step
        H_hat = self.H_hat/self.step
        H_hat_inv = np.linalg.inv(H_hat)
        self.V_hat = np.dot(np.dot(H_hat_inv, S_hat), H_hat_inv)/self.step
        self.se_hat = np.sqrt(np.diag(self.V_hat))
        
    def BatchMeansParameterVariance(self, K=None):
        if K is None:
            K = round(self.step**((1 - self.alpha)/2))
        K = max(K, 3)
        e = [round(pow((k + 1)/(K + 1), 1/(1 - self.alpha)) * self.step) 
             for k in range(K + 1)]
        V_hat = np.zeros((2 * p, 2 * p))
        e0 = e[0]
        b0 = self.beta_bar_log[e0 - 1]
        bK_bar = (self.beta_bar * self.step - b0 * e0)/(self.step - e0)
        for k in range(K):
            ek = e[k + 1]
            bk = self.beta_bar_log[ek - 1]
            bk_bar = (bk * ek - b0 * e0)/(ek - e0)
            m = bk_bar - bK_bar
            V_hat += (ek - e0) * np.outer(m, m)
            e0 = ek
            b0 = bk
        V_hat /= K
        return(V_hat)
    
    def PluginParameterVariance(self, use_IPW = True, return_SH = False):
        S_hat = np.zeros((2 * p, 2 * p))
        H_hat = np.zeros((2 * p, 2 * p))
        IPW = 1
        for t in range(self.step):
            O = self.data[t]
            beta = self.beta_bar_log[t]
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
        
    def DoubleWeighting(self, T, B = 100, explore = False):
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
            #self.beta_hat_log.append(self.beta_hat.copy())
            self.beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
            
    def ResamplingParameterVariance(self):
        return(np.cov(self.beta_bar_b, rowvar = False))

def eps(t):
    if t < 20:
        return(1)
    else:
        return(0.05)
    
def lr(t):
    return(0.5 * t**(-2/3))

LinearModel = model(mu = u, generator = LinearGenerator, loss = MSELoss,
                    eps = eps, alpha = 0.5, gamma = 2/3, lr = lr)

def MC(model, T, B, M):
    beta_bar = np.zeros((M, 2 * p))
    P_se_log = np.zeros((M, 2 * p))
    R_se_log = np.zeros((M, 2 * p))
    BM_se_log = np.zeros((M, 2 * p))
    P_cp = np.zeros(2 * p)
    R_cp = np.zeros(2 * p)
    BM_cp = np.zeros(2 * p)
    for i in range(M):
        model.Initialize()
        model.DoubleWeighting(T = T, B = B)
        beta_bar[i, :] = model.beta_bar
        P_se_i = np.sqrt(np.diag(model.PluginParameterVariance())/T)
        P_se_log[i, :] = P_se_i
        R_se_i = np.sqrt(np.diag(model.ResamplingParameterVariance()))
        R_se_log[i, :] = R_se_i
        BM_se_i = np.sqrt(np.diag(model.BatchMeansParameterVariance())/T)
        BM_se_log[i, :] = BM_se_i
        cover_P = np.abs(beta_true - model.beta_bar) <= 1.96 * P_se_i
        cover_R = np.abs(beta_true - model.beta_bar) <= 1.96 * R_se_i
        cover_BM = np.abs(beta_true - model.beta_bar) <= 1.96 * BM_se_i
        P_cp += np.array(cover_P, dtype=float)
        R_cp += np.array(cover_R, dtype=float)
        BM_cp += np.array(cover_BM, dtype=float)
    mcsd = np.std(beta_bar, axis = 0)
    P_se = np.mean(P_se_log, axis = 0)
    R_se = np.mean(R_se_log, axis = 0)
    BM_se = np.mean(BM_se_log, axis = 0)
    P_cp /= M
    R_cp /= M
    BM_cp /= M
    return(beta_bar, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp)

def MC1(model, T, B):
    np.random.seed()
    model.Initialize()
    model.DoubleWeighting(T = T, B = B)
    P_se = np.sqrt(np.diag(model.PluginParameterVariance())/T)
    R_se = np.sqrt(np.diag(model.ResamplingParameterVariance()))
    BM_se = np.sqrt(np.diag(model.BatchMeansParameterVariance())/T)
    P_cp = np.array(np.abs(beta_true - model.beta_bar) <= 1.96 * P_se, dtype=float)
    R_cp = np.array(np.abs(beta_true - model.beta_bar) <= 1.96 * R_se, dtype=float)
    BM_cp = np.array(np.abs(beta_true - model.beta_bar) <= 1.96 * BM_se, dtype=float)
    return(model.beta_bar, P_se, P_cp, R_se, R_cp, BM_se, BM_cp)

def ParallelMC(model, T, B, M, num_procs):
    pool = Pool(num_procs)
    _MC1_ = partial(MC1, model = model, T = T, B = B)
    experiments = [pool.apply_async(_MC1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    P_se_log = [e.get()[1] for e in experiments]
    P_cp_log = [e.get()[2] for e in experiments]
    R_se_log = [e.get()[3] for e in experiments]
    R_cp_log = [e.get()[4] for e in experiments]
    BM_se_log = [e.get()[5] for e in experiments]
    BM_cp_log = [e.get()[6] for e in experiments]
    mcsd = np.std(beta_bar, axis = 0)
    P_se = np.mean(P_se_log, axis = 0)
    P_cp = np.mean(P_cp_log, axis = 0)
    R_se = np.mean(R_se_log, axis = 0)
    R_cp = np.mean(R_cp_log, axis = 0)
    BM_se = np.mean(BM_se_log, axis = 0)
    BM_cp = np.mean(BM_cp_log, axis = 0)
    return(beta_bar, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp)

if __name__ == '__main__':
    print(' ' * 29 + 'b01  b02  b03  b11  b12  b13')
    for T in [1000, 10000, 100000]: #lr = 0.5*t^(-0.667)
        _, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp = ParallelMC(LinearModel, T = T, B = 200, M = 200, num_procs = 16)
#        _, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp = MC(LinearModel, T = T, B = 200, M = 200)
        print('PSE/MCSD at step {:8}:'.format(T), np.round(P_se/mcsd, 2))
        print('PluginCP at step {:8}:'.format(T), np.round(P_cp, 2))
        print('RSE/MCSD at step {:8}:'.format(T), np.round(R_se/mcsd, 2))
        print('ResampCP at step {:8}:'.format(T), np.round(R_cp, 2))
        print('BMSE/MCSD at step {:7}:'.format(T), np.round(BM_se/mcsd, 2))
        print('BtchMnCP at step {:8}:'.format(T), np.round(BM_cp, 2))
