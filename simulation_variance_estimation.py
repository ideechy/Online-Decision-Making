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
    return 1 if t <= 50 else 0.2


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
epsilon = 0.2
d = np.array(u(1, X, beta_true) > u(0, X, beta_true), dtype=float)
pi = (1 - epsilon) * d + epsilon / 2
S = block_diag(np.dot(X.T / (1 - pi), X), np.dot(X.T / pi, X)) / (4 * N) * sigma ** 2
H_inv = np.linalg.inv(H)
V = np.dot(np.dot(H_inv, S), H_inv)
val_true = np.mean(u(d, X, beta_true))
VV = (np.mean(u(d, X, beta_true) ** 2) + sigma ** 2) * 2 / (2 - epsilon) - val_true ** 2


# parallelize simulation functions
def MC1(model, T, R):
    np.random.seed(None)
    model.Initialize()
    model.DoubleWeighting(T=T, R=R)
    P_se = np.sqrt(np.diag(model.PluginParameterVariance()) / T)
    R_se = np.sqrt(np.diag(model.ResamplingParameterVariance()))
    BM_se = np.sqrt(np.diag(model.BatchMeansParameterVariance()) / T)
    P_cp = np.array(np.abs(np.asarray(beta_true) - model.beta_bar) <= 1.96 * P_se, dtype=float)
    R_cp = np.array(np.abs(np.asarray(beta_true) - model.beta_bar) <= 1.96 * R_se, dtype=float)
    BM_cp = np.array(np.abs(np.asarray(beta_true) - model.beta_bar) <= 1.96 * BM_se, dtype=float)
    return model.beta_bar, P_se, P_cp, R_se, R_cp, BM_se, BM_cp


def ParallelMC(model, T, R, M, num_procs):
    pool = Pool(num_procs)
    _MC1_ = partial(MC1, model=model, T=T, R=R)
    experiments = [pool.apply_async(_MC1_) for _ in range(M)]
    beta_bar = [e.get()[0] for e in experiments]
    P_se_log = [e.get()[1] for e in experiments]
    P_cp_log = [e.get()[2] for e in experiments]
    R_se_log = [e.get()[3] for e in experiments]
    R_cp_log = [e.get()[4] for e in experiments]
    BM_se_log = [e.get()[5] for e in experiments]
    BM_cp_log = [e.get()[6] for e in experiments]
    mcsd = np.std(beta_bar, axis=0)
    P_se = np.mean(P_se_log, axis=0)
    P_cp = np.mean(P_cp_log, axis=0)
    R_se = np.mean(R_se_log, axis=0)
    R_cp = np.mean(R_cp_log, axis=0)
    BM_se = np.mean(BM_se_log, axis=0)
    BM_cp = np.mean(BM_cp_log, axis=0)
    return beta_bar, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp


if __name__ == '__main__':
    LinearModel = ODMSGD(p=p, mu=u, generator=generator, loss=MSELoss,
                         eps=eps, eps_inf=epsilon, lr=lr, alpha=0.5, gamma=0.501)
    print(' ' * 28 + 'bet01 bet02 bet03 bet11 bet12 bet13')
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    pr = np.empty((3, 6))
    rr = np.empty((3, 6))
    br = np.empty((3, 6))
    pc = np.empty((3, 6))
    rc = np.empty((3, 6))
    bc = np.empty((3, 6))
    T = [1000, 10000, 100000]
    for i in range(len(T)):
        _, mcsd, P_se, P_cp, R_se, R_cp, BM_se, BM_cp = ParallelMC(LinearModel, T=T[i], R=200, M=5000, num_procs=16)
        pr[i, :] = P_se / mcsd
        rr[i, :] = R_se / mcsd
        br[i, :] = BM_se / mcsd
        pc[i, :] = P_cp
        rc[i, :] = R_cp
        bc[i, :] = BM_cp
        print('PSE/MCSD at step {:8}:'.format(T[i]), P_se / mcsd)
        print('PluginCP at step {:8}:'.format(T[i]), P_cp)
        print('RSE/MCSD at step {:8}:'.format(T[i]), R_se / mcsd)
        print('ResampCP at step {:8}:'.format(T[i]), R_cp)
        print('BMSE/MCSD at step {:7}:'.format(T[i]), BM_se / mcsd)
        print('BtchMnCP at step {:8}:'.format(T[i]), BM_cp)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ratio = np.array([pr, rr, br])
    coverage = np.array([pc, rc, bc])
    method = ['Plugin', 'Resampling', 'Batch-means']

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for j in range(6):
        for i in range(3):
            if j == 0:
                ax1.plot(np.log10(T), ratio[i][:, j], color=colors[i], label=method[i])
            else:
                ax1.plot(np.log10(T), ratio[i, :, j], color=colors[i])
            ax2.plot(np.log10(T), coverage[i, :, j], color=colors[i])
    ax1.set_xlabel("$\log_{10}$(decision steps)")
    ax1.set_ylabel("average SE/MCSD")
    ax2.set_xlabel("$\log_{10}$(decision steps)")
    ax2.set_ylabel("coverage probability")
    f.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", borderaxespad=0, ncol=3)

    plt.savefig("figure/linear_variance_estimation.pdf", bbox_inches='tight')
