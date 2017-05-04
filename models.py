#!/usr/bin/env python
#coding:utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(71)

class BernoulliBanditBase(object):
    def __init__(self, arm_num):
        self.arm_num = arm_num
        #self.history = np.zeros((self.arm_num, 2))
        self.params = np.zeros(self.arm_num)
        self.trials = np.zeros(self.arm_num)
        self.total_trial = 0
        self.history = []

    def ucb1(self):
        return self.params + np.sqrt(np.log(self.total_trial) / (2*self.trials) )

    def kl(self, p, q):
        return p *  np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

    def kl_grad(self, p, q):
        return (q-p)  / (q * (1-q))

    def kl_ucb(self, c = 0, delta = 1e-8, eps = 1e-12, max_iter = 1e2):
        upperbounds = (np.log(self.total_trial) + c * np.log(np.log(self.total_trial)) )/ self.trials
        upperbounds = np.maximum(delta, upperbounds)
        klucb_results = np.zeros(self.arm_num)
        for k in range(self.arm_num):
            p = self.params[k]
            if p >= 1:
                klucb_results[k] = 1
                continue
            q = p + delta
            for i in range(int(max_iter)):
                f = upperbounds[k] - self.kl(p,q)
                if ( f * f < eps):
                    break
                df = - self.kl_grad(p,q)
                q = min(1 - delta, max(q - f / df, p + delta))
            klucb_results[k] = q
        return klucb_results

    def thompson_sampling(self, prior_alpha, prior_beta):
        ts_results = np.zeros(self.arm_num)
        for k in range(self.arm_num):
            alpha_k = self.trials[k] * self.params[k]
            ts_results[k] = np.random.beta(alpha_k + prior_alpha, self.trials[k] - alpha_k + prior_beta)
        return ts_results

    def random_arm(self):
        return np.random.choice(self.arm_num)

    def update_one_arm(self, index, reward):
        #self.history[index] += 1
        self.params[index] = ( self.trials[index] * self.params[index] + reward ) / (self.trials[index] + 1)
        self.trials[index] += 1
        self.total_trial += 1

class BernoulliBandit(BernoulliBanditBase):
    def __init__(self, arm_num, strategy, eps_greedy = 0.1, c = 0, delta = 1e-12, eps = 1e-8, max_iter = 1e3
        , prior_alpha = 0.5, prior_beta = 0.5):
        super(BernoulliBandit, self).__init__(arm_num)
        self.strategy = strategy
        self.eps_greedy = eps_greedy
        self.c = c
        self.eps = eps
        self.delta = delta
        self.max_iter = max_iter
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self):
        if self.total_trial == 0:
            return self.random_arm()

        if self.strategy == "egreedy":
            random_search_flg = np.random.binomial(1, self.eps_greedy) #np.random.uniform < self.eps_greedy
            if random_search_flg:
                return self.random_arm()
            else:
                return np.argmax(self.params)

        elif self.strategy =="ucb1":
            ucb1_results = self.ucb1()
            return np.argmax(ucb1_results)

        elif self.strategy == "kl_ucb":
            klucb_results = self.kl_ucb(self.c, self.delta, self.eps, self.max_iter)
            return np.argmax(klucb_results)

        elif self.strategy == "ts":
            ts_results = self.thompson_sampling(self.prior_alpha, self.prior_beta)
            return np.argmax(ts_results)

        else:
            print("Sorry, the strategy is not defined yet.")
            return None

    def save_history(self, index, reward):
        self.history.append([index,reward])

    def update_arm(self):
        self.update_one_arm(self.history[-1][0], self.history[-1][1])

    def print_params(self):
        print(self.strategy + "params ={}".format(self.params))


def tmp(arm):
    # temporary reward function
    return np.random.binomial(1, true_params[arm])

def get_regret():
    indices = np.array(agent.history)[:,0]
    expected_rewards = true_params[indices]
    best_reward = np.max(true_params)
    regrets = best_reward - expected_rewards
    return regrets



# simple model test
arm_num = 3
true_params = np.random.uniform(0.05, 0.2 ,size = arm_num)
#true_params = np.arange(arm_num) / np.arange(arm_num)
strategy_list = ["egreedy", "ucb1", "ts","kl_ucb"]

print("true_params = {}".format(true_params))

for strategy in strategy_list:
    agent = BernoulliBandit( arm_num, strategy)
    t = 0
    while t < 1e5:
        arm = agent.select_arm()
        reward = tmp(arm)
        agent.save_history(arm, reward)
        agent.update_arm()
        t += 1
    agent.print_params()

    regrets = get_regret()
    cum_regrets = np.cumsum(regrets)
    np.save("regret_" + strategy , cum_regrets)
    plt.plot(cum_regrets, label = strategy)
plt.legend(loc="best")
plt.savefig("figures/regret_bernoulli.png")



