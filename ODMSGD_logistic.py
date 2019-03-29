#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from scipy.linalg import block_diag
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
    
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def logit01(A, X, beta):
    return(sigmoid(u(A, X, beta)))

def gu(A, X, beta):
    return(np.vstack(((1 - A) * X.T, A * X.T)))

def LogisticGenerator01(n=1, X=None, A=None, generate_Y=False):
    if X is None:
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1:] = np.random.normal(0, 1, size = (n, (p - 1)))
    else:
        assert X.shape[1] == p
        n = X.shape[0]
    if not generate_Y:
        return(X, A)
    elif A is not None:
        # if A not 0 or 1, raise error
        # if size of A does not match X.shape[0], raise error
        Y = np.random.binomial(1, logit01(A, X, beta_true))
        return((X, A, Y))
    else:
        Y0 = np.random.binomial(1, logit01(0, X, beta_true))
        Y1 = np.random.binomial(1, logit01(1, X, beta_true))
        Y = {'0': Y0, '1': Y1}
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
                 lr = None, alpha = 1, gamma = 2/3):
        self.mu = mu
        self.generator = generator
        self.loss = loss
        self.eps_inf = eps_inf
        if eps is None:
            self.eps = lambda x: eps_inf
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
        self.se_hat = np.zeros(2 * p)
        # initialization for value estimation
        self.val_hat = 0.
        self.cum_Y = 0.
        self.cum_Ysq = 0.
        self.vse_hat = 0.
        
    def SGD(self, T, use_IPW = True, explore = False, real_data = False):
        for t in range(self.step, self.step + T):
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
                        data.loc[data[data.match == 0].index[0], 'match'] = -1
            # observe Y_t
            O = self.generator(X = X, A = A, generate_Y = True)
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

def MCPlugin1(model, T):
    np.random.seed()
    model.Initialize()
    model.SGD(T = T)
    P_cp = np.array(np.abs(beta_true - model.beta_bar) <= 1.96 * model.se_hat, dtype=float)
    V_cp = np.array(np.abs(val_true - model.val_hat) <= 1.96 * model.vse_hat, dtype=float)
    return(model.beta_bar, model.val_hat, model.se_hat, P_cp, model.vse_hat, V_cp)

def ParallelMCPlugin(model, T, M, num_procs):
    pool = Pool(num_procs)
    _MCPlugin1_ = partial(MCPlugin1, model = model, T = T)
    experiments = [pool.apply_async(_MCPlugin1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    val_hat = [e.get()[1] for e in experiments]
    P_se_log = [e.get()[2] for e in experiments]
    P_cp_log = [e.get()[3] for e in experiments]
    V_se_log = [e.get()[4] for e in experiments]
    V_cp_log = [e.get()[5] for e in experiments]
    P_mcsd = np.std(beta_bar, axis = 0)
    P_se = np.mean(P_se_log, axis = 0)
    P_cp = np.mean(P_cp_log, axis = 0)
    V_mcsd = np.std(val_hat, axis = 0)
    V_se = np.mean(V_se_log, axis = 0)
    V_cp = np.mean(V_cp_log, axis = 0)
    return(beta_bar, val_hat, P_mcsd, P_se, P_cp, V_mcsd, V_se, V_cp)

def eps(t):
    if t < 50:
        return(1)
    else:
        return(0.2)

def lr(t):
    return(0.5 * t**(-0.501))

N = 1000000
X, _ = LogisticGenerator01(N)
A = np.random.binomial(1, 0.5, N)
O = LogisticGenerator01(N, X, A, True)
_, H = BCELoss(beta_true, O, return_second_moment=True)
epsilon = 0.2
mu0 = logit01(0, X, beta_true)
mu1 = logit01(1, X, beta_true)
d = np.array(mu1 > mu0, dtype = float)
pi = (1 - epsilon) * d + epsilon/2
from scipy.linalg import block_diag
S = block_diag(np.dot(X.T * (mu0 - mu0**2)/(1-pi), X), np.dot(X.T * (mu1 - mu1**2)/pi, X))/(4 * N)
H_inv = np.linalg.inv(H)
V = np.dot(np.dot(H_inv, S), H_inv)
val_true = np.mean(logit01(d, X, beta_true))
VV = val_true * 2/(2 - epsilon) - val_true**2

if __name__ == '__main__':
    LogisticModel01 = model(mu = logit01, generator = LogisticGenerator01, loss = BCELoss,
                            eps = eps, eps_inf = 0.2, lr = lr, alpha = 0.5, gamma = 0.501)
    print(' ' * 25 + '[beta01 beta02 beta03 beta11 beta12 beta13]  value')
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for T in [1000, 10000, 100000]:
        beta_bar, val_hat, P_mcsd, P_se, P_cp, V_mcsd, V_se, V_cp = ParallelMCPlugin(LogisticModel01, T = T, M = 5000, num_procs = 16)
        print('ASE/MCSD at step {:6}:'.format(T), P_se/P_mcsd, '{:0.4f}'.format(V_se/V_mcsd))
        print('CvrgProb at step {:6}:'.format(T), P_cp, '{:0.4f}'.format(V_cp))
