#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import t

def u(A, X, beta):
    if type(A) == int:
        if A == 1:
            return(X.dot(beta[p:]))
        else:
            return(X.dot(beta[:p]))
    else:
        return((1 - A) * X.dot(beta[:p]) + A * X.dot(beta[p:]))
    
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def logit01(A, X, beta):
    return(sigmoid(u(A, X, beta)))

def gu(A, X, beta):
    return(np.vstack(((1 - A) * X.T, A * X.T)))

def RealGenerator(n=1, X=None, A=None, generate_Y=False):
    global pointer
    df = data.iloc[pointer]
    if X is None and A is None:
        X = df.iloc[2:2+p].to_numpy().reshape(1, -1)
        A = df.iat[1]
    if not generate_Y:
        return(X, A)
    else:
        Y = df.iat[0]
        pointer += 1
        return((X, A, Y))
    
def BCELoss(beta, O, gradient_only = False, return_second_moment = False):
    X, A, Y = O
    n = X.shape[0]
    mu = logit01(A, X, beta)
    if not return_second_moment:
        grad = np.dot(gu(A, X, beta), mu - Y)/n
        if gradient_only:
            return(grad)
        else:
            l = np.mean(-Y * np.log(mu) - (1 - Y) * np.log(1 - mu))
        return((l, grad))
    else:
        gu_v = gu(A, X, beta)
        grad_v = gu_v * (mu - Y)
        Sigma = grad_v.dot(grad_v.T)/n
        Hessian = np.dot(gu_v * mu * (1 - mu), gu_v.T)/n
        return((Sigma, Hessian))
    
class model:
    def __init__(self, mu, generator, loss, 
                 eps = None, eps_inf = 0.01, 
                 alpha = 1, gamma = 2/3):
        self.mu = mu
        self.generator = generator
        self.loss = loss
        if eps is None:
            self.eps = lambda x: eps_inf
        else:
            self.eps = eps
        self.eps_inf = eps_inf
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lambda x: alpha * x**(-gamma)
        
    def Initialize(self):
        self.beta_hat = np.zeros(2 * p)
        self.beta_bar = np.zeros(2 * p)
        #self.beta_hat_log = [self.beta_hat.copy()] # start from step 1
        self.beta_bar_log = [self.beta_bar.copy()] # start from step 1
        self.loss_log = []
        self.step = 0
        # initialization for plugin variance estimation
        self.S_hat = np.zeros((2 * p, 2 * p))
        self.H_hat = np.zeros((2 * p, 2 * p))
        self.se_hat = np.zeros(2 * p)
        # initialization for value estimation
        self.val_hat = 0.
        self.cum_Y = 0.
        self.cum_Ysq = 0.
        self.vse_hat = 0.
        # initialization for DoubleWeighting method
        self.beta_hat_b = None
        self.beta_bar_b = None
        
    def SGD(self, T, use_IPW = True, explore = False, real_data = False):
        global pointer
        for t in range(self.step, self.step + T):
            if t%10000 == 0:
                print('Iteration Step{:7}: value {:6.4f}'.format(t, self.val_hat))
            match = False
            while not match:
                # observe X_t
                X, real_A = self.generator()
                if explore:
                    epsilon = 1
                    d = 0
                    self.pi_hat = 0.5
                else:
                    # update pi_hat_t-1(X_t)
                    epsilon = self.eps(t)
                    d = int(self.mu(1, X, self.beta_bar) > self.mu(0, X, self.beta_bar))
                    self.pi_hat = (1 - epsilon) * d + epsilon/2 
                # sample A_t from Bernoulli pi_hat
                A = np.random.binomial(1, self.pi_hat)
                if not real_data:
                    match = True
                else:
                    match = (A == real_A)
                    if not match:
                        pointer += 1
                        if pointer == N:
                            break
            # observe Y_t
            if not match:
                break
            O = self.generator(X = X, A = A, generate_Y = True)
            if pointer == N:
                break
            Y = O[2].squeeze()
            # update treatment consistency
            C = int(A == d)
            pi_C = 1 - epsilon/2
            # update value estimation
            self.cum_Y += C * Y/pi_C
            self.cum_Ysq += C * Y**2/pi_C
            self.val_hat = self.cum_Y/(t + 1)
            # calculate IPW
            if use_IPW:
                IPW = (A/self.pi_hat + (1 - A)/(1 - self.pi_hat))/2
            else:
                IPW = 1
            # update second moment estimation
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
            # log historical info
            self.beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1
        S_hat = self.S_hat/self.step
        H_hat = self.H_hat/self.step
        H_hat_inv = np.linalg.inv(H_hat)
        V_hat = np.dot(np.dot(H_hat_inv, S_hat), H_hat_inv)
        self.se_hat = np.sqrt(np.diag(V_hat)/self.step)
        self.vse_hat = np.sqrt((self.cum_Ysq/self.step * 2/(2 - self.eps_inf) - 
                                self.val_hat**2)/self.step)
        
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

def eps(t):
    if t < 50:
        return(1)
    else:
        return(0.2)
        #return(min(1, 0.1 * np.log(t)/np.sqrt(t)))
        
data = pd.read_csv("yahoo.csv")
p = 5
N = data.shape[0]
pointer = 0
np.random.seed(1)
    
YahooLogistic = model(mu = logit01, generator = RealGenerator, loss = BCELoss,
                      eps = eps, eps_inf = 0.2, alpha = 0.5, gamma = 0.501)
YahooLogistic.Initialize()
YahooLogistic.SGD(T = 500000, real_data = True)
print('Data used: {}, data matched: {}'.format(pointer, YahooLogistic.step))
print('Parameter estimation:')
print(' ' * 7 + 'estimate    s.e. [  95% Wald CI] t value P(>|t|)')
for i in [0,1]:
    for j in range(p):
        est = YahooLogistic.beta_bar[i * p + j]
        se = YahooLogistic.se_hat[i * p + j]
        lo = est - 1.96 * se
        up = est + 1.96 * se
        tv = est/se
        pv = 2 * t.sf(abs(tv), YahooLogistic.step - 2 * p)
        print('beta' + str(i) + str(j + 1) + 
              '{:9.4f}{:8.4f}{:8.4f}{:8.4f}{:8.2f}{:8.4f}'.format(est, se, lo, up, tv, pv))
print('Value estimation:')
print('value {:9.4f}{:8.4f}'.format(YahooLogistic.val_hat, YahooLogistic.vse_hat) + 
     '{:8.4f}{:8.4f}'.format(YahooLogistic.val_hat - 1.96 * YahooLogistic.vse_hat, 
                             YahooLogistic.val_hat + 1.96 * YahooLogistic.vse_hat) +
     '{:8.2f}{:8.4f}'.format(YahooLogistic.val_hat/YahooLogistic.vse_hat, 
                             2 * t.sf(abs(YahooLogistic.val_hat/YahooLogistic.vse_hat), YahooLogistic.step - 2 * p)))