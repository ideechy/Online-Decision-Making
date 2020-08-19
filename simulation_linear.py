#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from multiprocessing import Pool
from functools import partial
from ODMSGD import ODMSGD
from utils import u, LinearGenerator, MSELoss

p = 3
sigma = 0.1
beta_true = [0.3, -0.1, 0.7, 0.8, 0.5, -0.4]


def eps(t):
    return 1 if t <= 50 else max(t**(-0.3), 0.1)


def lr(t):
    return 0.5 * t ** (-0.501)


def generator(n=1, X=None, A=None, generate_Y=False):
    return LinearGenerator(beta_true, sigma, n, X, A, generate_Y)


# approximate the population value using a large sample
N = 1000000
X, _ = generator(N)
A = np.random.binomial(1, 0.5, N)
O = generator(N, X, A, True)
_, H = MSELoss(beta_true, O, return_second_moment=True)
epsilon = 0.1
d = np.array(u(1, X, beta_true) > u(0, X, beta_true), dtype=float)
pi = (1 - epsilon) * d + epsilon/2
S = block_diag(np.dot(X.T/(1-pi), X), np.dot(X.T/pi, X))/(4 * N) * sigma**2
H_inv = np.linalg.inv(H)
V = np.dot(np.dot(H_inv, S), H_inv)
val_true = np.mean(u(d, X, beta_true))
VV = (np.mean(u(d, X, beta_true)**2) + sigma**2) * 2/(2 - epsilon) - val_true**2


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
    V_mcsd = np.nanstd(val_hat, axis=0)
    V_se = np.nanmean(V_se_log, axis=0)
    V_cp = np.mean(V_cp_log, axis=0)
    return beta_bar, val_hat, P_mcsd, P_se, P_cp, V_mcsd, V_se, V_cp


def ParallelMCPluginSeq(model, T, M, num_procs):
    pool = Pool(num_procs)
    _MCPluginSeq1_ = partial(MCPluginSeq1, model = model, T = T)
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
    LinearModel = ODMSGD(p=p, mu=u, generator=generator, loss=MSELoss,
                         eps=eps, eps_inf=epsilon, lr=lr, alpha=0.5, gamma=0.501)
    T = 1000
    beta_bar_m, beta_bar_u, beta_bar_l, val_hat_m, val_hat_u, val_hat_l = \
        ParallelMCPluginSeq(LinearModel, T=T, M=5000, num_procs=8)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

    for i in range(2 * p):
        ax1.plot(beta_bar_m[:, i], lw=1, color=colors[i])
        ax1.fill_between(np.arange(T), beta_bar_l[:, i], beta_bar_u[:, i], color=colors[i], alpha=0.4)
        ax1.scatter(T * 1.05, beta_true[i], color=colors[i])
        ax1.set_xlabel("decision steps")
        ax1.set_ylabel("parameter")

    ax2.plot(val_hat_m, lw=1, color=colors[0])
    ax2.fill_between(np.arange(T), val_hat_l, val_hat_u, color=colors[0], alpha=0.4)
    ax2.scatter(T * 1.05, val_true, color=colors[0])
    ax2.set_xlabel("decision steps")
    ax2.set_ylabel("value")

    plt.savefig("figure/linear_estimator_convergence.pdf")
