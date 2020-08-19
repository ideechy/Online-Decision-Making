#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.linalg import block_diag
from ODMSGD import ODMSGD
from utils import logit01, LogisticGenerator01, BCELoss

p = 3
beta_true = [0.3, -0.1, 0.7, 0.8, 0.5, -0.4]


def eps(t):
    return 1 if t <= 50 else max(t ** (-0.3), 0.1)


def lr(t):
    return 0.5 * t ** (-0.501)


def generator(n=1, X=None, A=None, generate_Y=False):
    return LogisticGenerator01(beta_true, n, X, A, generate_Y)


# approximate the population value using a large sample
N = 1000000
X, _ = generator(N)
A = np.random.binomial(1, 0.5, N)
O = generator(N, X, A, True)
_, H = BCELoss(beta_true, O, return_second_moment=True)
epsilon = 0.1
mu0 = logit01(0, X, beta_true)
mu1 = logit01(1, X, beta_true)
d = np.array(mu1 > mu0, dtype=float)
pi = (1 - epsilon) * d + epsilon / 2
S = block_diag(np.dot(X.T * (mu0 - mu0 ** 2) / (1 - pi), X), np.dot(X.T * (mu1 - mu1 ** 2) / pi, X)) / (4 * N)
H_inv = np.linalg.inv(H)
V = np.dot(np.dot(H_inv, S), H_inv)
val_true = np.mean(logit01(d, X, beta_true))
VV = val_true * 2 / (2 - epsilon) - val_true ** 2


# parallelize simulation functions
def MCPlugin1(model, T):
    np.random.seed(None)
    model.Initialize()
    model.SGD(T=T)
    P_cp = np.array(np.abs(np.asarray(beta_true) - model.beta_bar) <= 1.96 * model.se_hat, dtype=float)
    V_cp = np.array(np.abs(val_true - model.val_hat) <= 1.96 * model.vse_hat, dtype=float)
    return model.beta_bar, model.val_hat, model.se_hat, P_cp, model.vse_hat, V_cp


def MCPluginSeq1(model, T):
    np.random.seed(None)
    model.Initialize()
    model.SGD(T=T)
    return model.beta_bar_log[1:], model.val_hat_log


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
    P_mcsd = np.std(beta_bar, axis=0)
    P_se = np.mean(P_se_log, axis=0)
    P_cp = np.mean(P_cp_log, axis=0)
    V_mcsd = np.std(val_hat, axis=0)
    V_se = np.mean(V_se_log, axis=0)
    V_cp = np.mean(V_cp_log, axis=0)
    return beta_bar, val_hat, P_mcsd, P_se, P_cp, V_mcsd, V_se, V_cp


def ParallelMCPluginSeq(model, T, M, num_procs):
    pool = Pool(num_procs)
    _MCPluginSeq1_ = partial(MCPluginSeq1, model=model, T=T)
    experiments = [pool.apply_async(_MCPluginSeq1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    val_hat = [e.get()[1] for e in experiments]
    beta_bar_m = np.mean(beta_bar, axis=0)
    beta_bar_u = np.quantile(beta_bar, q=0.975, axis=0)
    beta_bar_l = np.quantile(beta_bar, q=0.025, axis=0)
    val_hat_m = np.mean(val_hat, axis=0)
    val_hat_u = np.quantile(val_hat, q=0.975, axis=0)
    val_hat_l = np.quantile(val_hat, q=0.025, axis=0)
    return beta_bar_m, beta_bar_u, beta_bar_l, val_hat_m, val_hat_u, val_hat_l


if __name__ == '__main__':
    # initialize the model
    LogisticModel01 = ODMSGD(p=p, mu=logit01, generator=generator, loss=BCELoss,
                             eps=eps, eps_inf=epsilon, lr=lr, alpha=0.5, gamma=0.501)
    print(' ' * 25 + '[bet01 bet02 bet03 bet11 bet12 bet13] value')
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    for T in [1000, 10000, 100000]:
        beta_bar, val_hat, P_mcsd, P_se, P_cp, V_mcsd, V_se, V_cp = \
            ParallelMCPlugin(LogisticModel01, T=T, M=5000, num_procs=16)
        print('ASE/MCSD at step {:6}:'.format(T), P_se/P_mcsd, '{:0.3f}'.format(V_se/V_mcsd))
        print('CvrgProb at step {:6}:'.format(T), P_cp, '{:0.3f}'.format(V_cp))
        print('AvgCILen at step {:6}:'.format(T), 3.92 * P_se, '{:0.3f}'.format(3.92 * V_cp))
