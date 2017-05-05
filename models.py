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

    def ucb(self):
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
    def __init__(self, arm_num, strategy, eps_greedy = 0.01, c = 0, delta = 1e-12, eps = 1e-8, max_iter = 1e3
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

        elif self.strategy =="ucb":
            ucb_results = self.ucb()
            return np.argmax(ucb_results)

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


def tmp_reward(arm):
    # temporary reward function
    return np.random.binomial(1, true_params[arm])

def get_regret():
    indices = np.array(agent.history)[:,0]
    expected_rewards = true_params[indices]
    best_reward = np.max(true_params)
    regrets = best_reward - expected_rewards
    return regrets

def get_asymptoic_lower_bound(mode):
    if mode == "simple":
        return [0.1*(arm_num - 1)/agent.kl(0.4,0.5) * (np.log(t) - np.log(100)) for t in range(int(T))]
    else:
        return [sum([(true_params[0] - true_params[k]) * (np.log(t) - np.log(100))/ (agent.kl(true_params[k], true_params[0]) ) for k in range(1,arm_num)]) for t in range(int(T))]

def visualize_regret(mode):
    max_num = 0
    for strategy in strategy_list:
        cum_regrets = np.load("results/regret_" + strategy + ".npy")
        plt.plot(cum_regrets, label = strategy)
        max_num = max(max_num, max(cum_regrets))
    #asymptoic_lower_bounds = [sum([(true_params[0] - true_params[k]) * (np.log(t) - np.log(100))/ (kl(true_params[k], true_params[0]) ) for k in range(1,arm_num)]) for t in range(1, int(T))]
    #ab=[0.1*99/kl(0.4,0.5) * (np.log(t) - np.log(100)) for t in range(int(T))]
    ab = get_asymptoic_lower_bound(mode)
    max_num = max(max(ab), max_num)
    plt.plot(ab, label = "asymptoic bound")
    #plt.xticks([1e2,1e3,1e4,1e5], ["1e2", "1e3", "1e4", "1e5"])
    xticks_name = ["1e" + str(t) for t in range(2, int(np.log10(T)))]
    xticks = [float(xtick) for xtick in xticks_name]
    plt.xticks(xticks, xticks_name)
    plt.xlim(xmin = 100)
    #plt.ylim(0,10000)
    plt.ylim(0,max_num*2)
    plt.xscale("log")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig("figures/regret_bernoulli.png")

# simple model test
T = 1e4
arm_num = 10
#true_params = np.random.uniform(0.05, 0.2 ,size = arm_num)
#true_params = 0.1 + np.arange(arm_num) / (arm_num * 5.0)
true_params = np.ones(arm_num) * 0.4
true_params[0] += 0.1
strategy_list = ["egreedy", "ucb", "ts","kl_ucb"]
#eps_greedy_optimal = (2.0 * arm_num * np.log(T) ) / ( 0.005 * T)

print("true_params = {}".format(true_params))

for strategy in strategy_list:
    agent = BernoulliBandit( arm_num, strategy)
    t = 0
    while t < T:
        arm = agent.select_arm()
        reward = tmp_reward(arm)
        agent.save_history(arm, reward)
        agent.update_arm()
        t += 1
    agent.print_params()

    regrets = get_regret()
    cum_regrets = np.cumsum(regrets)
    np.save("results/regret_" + strategy , cum_regrets)
    #plt.plot(cum_regrets, label = strategy)

visualize_regret("simple")


