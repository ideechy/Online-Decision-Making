#!/usr/bin/env python3
# coding: utf-8

import numpy as np


class ODMSGD:
    def __init__(self, p, mu, generator, loss,
                 eps=None, eps_inf=0.01,
                 lr=None, alpha=1, gamma=2 / 3):
        self.p = p
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
            self.lr = lambda x: alpha * x ** (-gamma)
        else:
            self.lr = lr

    def Initialize(self):
        self.beta_hat = np.zeros(2 * self.p)
        self.beta_bar = np.zeros(2 * self.p)
        self.data = []
        self.pi_log = []  # start from step 0
        # self.beta_hat_log = [self.beta_hat.copy()] # start from step 1
        self.beta_bar_log = [self.beta_bar.copy()]  # start from step 1
        self.loss_log = []
        self.step = 0
        # initialization for plugin variance estimation
        self.S_hat = np.zeros((2 * self.p, 2 * self.p))
        self.H_hat = np.zeros((2 * self.p, 2 * self.p))
        self.se_hat = np.zeros(2 * self.p)
        # initialization for value estimation
        self.val_hat = 0.
        self.val_hat_log = []
        self.cum_Y = 0.
        self.cum_Ysq = 0.
        self.vse_hat = 0.
        # initialization for DoubleWeighting method
        self.beta_hat_b = None
        self.beta_bar_b = None

    def SGD(self, T, use_IPW=True, explore=False, real_data=False, data=None):
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
                    epsilon = self.eps(t + 1)
                    d = int(self.mu(1, X, self.beta_bar) > self.mu(0, X, self.beta_bar))
                    self.pi_hat = (1 - epsilon) * d + epsilon / 2
                    # sample A_t from Bernoulli pi_hat
                A = np.random.binomial(1, self.pi_hat)
                if not real_data:
                    match = True
                else:
                    match = (A == real_A)
                    if not match:
                        data.loc[data[data.match == 0].index[0], 'match'] = -1
            # observe Y_t
            O = self.generator(X=X, A=A, generate_Y=True)
            Y = O[2].squeeze()
            # update treatment consistency
            C = int(A == d)
            pi_C = 1 - epsilon / 2
            # update value estimation
            self.cum_Y += C * Y / pi_C
            self.cum_Ysq += C * Y ** 2 / pi_C
            self.val_hat = self.cum_Y / (t + 1)
            # calculate IPW
            if use_IPW:
                IPW = (A / self.pi_hat + (1 - A) / (1 - self.pi_hat)) / 2
            else:
                IPW = 1
            # update second moment estimation
            S, H = self.loss(self.beta_bar, O, return_second_moment=True)
            self.S_hat += S * IPW ** 2
            self.H_hat += H * IPW
            # update beta_hat_t
            learning_rate = self.lr(t + 1)
            l, g = self.loss(self.beta_hat, O)
            g *= IPW
            self.beta_hat -= learning_rate * g
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar) / (t + 1)
            # log historical info, need this if using PluginParameterVariance
            # self.data.append(O)
            # self.pi_log.append(self.pi_hat)
            # self.beta_hat_log.append(self.beta_hat.copy())
            self.beta_bar_log.append(self.beta_bar.copy())
            self.val_hat_log.append(self.val_hat)
            self.loss_log.append(l)
            self.step = t + 1
        S_hat = self.S_hat / self.step
        H_hat = self.H_hat / self.step
        H_hat_inv = np.linalg.inv(H_hat)
        V_hat = np.dot(np.dot(H_hat_inv, S_hat), H_hat_inv)
        self.se_hat = np.sqrt(np.diag(V_hat) / self.step)
        self.vse_hat = np.sqrt((self.cum_Ysq / self.step * 2 / (2 - self.eps_inf) -
                                self.val_hat ** 2) / self.step)

    def BatchMeansParameterVariance(self, K=None):
        if K is None:
            K = round(self.step ** ((1 - self.alpha) / 2))
        K = max(K, 3)
        e = [round(pow((k + 1) / (K + 1), 1 / (1 - self.alpha)) * self.step)
             for k in range(K + 1)]
        V_hat = np.zeros((2 * self.p, 2 * self.p))
        e0 = e[0]
        b0 = self.beta_bar_log[e0 - 1]
        bK_bar = (self.beta_bar * self.step - b0 * e0) / (self.step - e0)
        for k in range(K):
            ek = e[k + 1]
            bk = self.beta_bar_log[ek - 1]
            bk_bar = (bk * ek - b0 * e0) / (ek - e0)
            m = bk_bar - bK_bar
            V_hat += (ek - e0) * np.outer(m, m)
            e0 = ek
            b0 = bk
        V_hat /= K
        return (V_hat)

    def PluginParameterVariance(self, use_IPW=True, return_SH=False):
        S_hat = np.zeros((2 * self.p, 2 * self.p))
        H_hat = np.zeros((2 * self.p, 2 * self.p))
        IPW = 1
        for t in range(self.step):
            O = self.data[t]
            beta = self.beta_bar_log[t]
            Sigma, Hessian = self.loss(beta, O, return_second_moment=True)
            if use_IPW:
                A = O[1]
                IPW = (A / self.pi_log[t] + (1 - A) / (1 - self.pi_log[t])) / 2
            S_hat += Sigma * IPW ** 2
            H_hat += Hessian * IPW
        S_hat /= self.step
        H_hat /= self.step
        H_hat_inv = np.linalg.inv(H_hat)
        V_hat = np.dot(np.dot(H_hat_inv, S_hat), H_hat_inv)
        if return_SH:
            return S_hat, H_hat
        else:
            return V_hat

    def PluginValueVariance(self):
        beta = self.beta_bar
        mu_max = 0.
        thetasq = 0.
        for t in range(self.step):
            O = self.data[t]
            X, _, __ = O
            mu0 = self.mu(0, X, beta).squeeze()
            mu1 = self.mu(1, X, beta).squeeze()
            if mu1 > mu0:
                mu_max += mu1
                # thetasq += mu1**2 + sigma**2
            else:
                mu_max += mu0
                # thetasq += mu0**2 + sigma**2
            thetasq += 1
        etasq = thetasq / self.step * 2 / (2 - self.eps_inf) - (mu_max / self.step) ** 2
        return (float(etasq))

    def DoubleWeighting(self, T, R=100, explore=False, real_data=False, data=None):
        if self.beta_hat_b is None:
            self.beta_hat_b = np.zeros((R, 2 * self.p))
        if self.beta_bar_b is None:
            self.beta_bar_b = np.zeros((R, 2 * self.p))
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
                    self.pi_hat = (1 - epsilon) * d + epsilon / 2
                    # sample A_t from Bernoulli pi_hat
                A = np.random.binomial(1, self.pi_hat)
                if not real_data:
                    match = True
                else:
                    match = (A == real_A)
                    if not match:
                        data.loc[data[data.match == 0].index[0], 'match'] = -1
            # observe Y_t
            O = self.generator(X=X, A=A, generate_Y=True)
            # calculate gradient
            learning_rate = self.lr(t + 1)
            W_ip = (A / self.pi_hat + (1 - A) / (1 - self.pi_hat)) / 2
            l, g = self.loss(self.beta_hat, O)
            # update beta_hat_t
            self.beta_hat -= learning_rate * W_ip * g
            # update beta_bar_t
            self.beta_bar = (self.beta_hat + t * self.beta_bar) / (t + 1)
            # Resampling
            W_b = np.random.exponential(size=(R, 1))
            G = np.array([self.loss(b, O, gradient_only=True) for b in self.beta_hat_b])
            self.beta_hat_b -= learning_rate * W_ip * W_b * G
            self.beta_bar_b = (self.beta_hat_b + t * self.beta_bar_b) / (t + 1)
            # log
            self.data.append(O)
            self.pi_log.append(self.pi_hat)
            # self.beta_hat_log.append(self.beta_hat.copy())
            self.beta_bar_log.append(self.beta_bar.copy())
            self.loss_log.append(l)
            self.step = t + 1

    def ResamplingParameterVariance(self):
        return np.cov(self.beta_bar_b, rowvar=False)

