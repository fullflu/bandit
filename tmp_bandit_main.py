#!/usr/bin/env python
#coding:utf-8
#fullflu

# now (2016/02/23) three bandit algorithms can be implemented (this code is not completed , so will be modified and other algorithms will be added)
  # 1 : Thompson sampling with online logistic regression (Laplace approximation) , based on Algorithm 3 at Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).
  # 2 : Epsilon greedy action selection with online logistic regression (sklearn.linear_model.SGDClassifier)
  # 3 : Random action selection with online logistic regression (sklearn.linear_modle.SGDClassifier)

# I generated sample input and reward
# I considered interaction between features (included action fields)
# Online logisic regression of SGDClassifier will be updated to Laplace approximation
# Evaluation score will be calculated by regret

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier
import time
np.random.seed(71)

class Thompson_logistic(object):
    def __init__(self,x_dim,dim,arm_num,lam=0.1,w_true=None):
        self.x_dim = x_dim
        self.arm_num = arm_num
        self.dim = dim
        self.lam = lam
        self.m = np.zeros(dim)
        self.q = np.ones(dim) * lam
        if w_true == None:
            self.w_true = np.random.normal(size=dim)
        else:
            self.w_true = w_true

    def sample_weight(self):
        self.w = np.random.normal(self.m,self.q ** (-1/2.),size = self.dim)
        #return self.w

    def get_loss(self,w,*args):
        X,y = args
        loss = 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1+np.exp(-1 * y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        return loss

    def get_grad(self,w,*args):
        X,y = args
        return self.q * (w - self.m) + -1 * np.array([y[j] *  X[j] * np.exp(-1 * y[j] * w.dot(X[j])) / (1. + np.exp(-1 * y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0) #np.array([y[j] *  X[j] / (1. + np.exp(-1 * y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    def update(self,X,y):
        self.m = minimize(self.get_loss,self.w,args=(X,y)).x#,jac=self.get_grad,method="BFGS").x
        P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)

    def update_with_grad(self,X,y):
        self.m = minimize(self.get_loss,self.w,args=(X,y),jac=self.get_grad,method="BFGS").x
        P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)

    def predict_proba(self,X,w=None):
        if w == None:
            proba = 1 / (1 + np.exp(-1 * X.dot(self.w)))
            return np.array([1-proba , proba]).T
        else:
            proba = 1 / (1 + np.exp(-1 * X.dot(w)))
            return np.array([1-proba , proba]).T

    def predict(self,X,w=None):
        proba = self.predict_proba(X,w)
        if proba.ndim > 1:
            proba = proba[:,1]
        else:
            proba = proba[1]
        pred = np.zeros_like(proba)
        pred[proba >= 0.5] = 1.
        return pred

    def select_arm(self,x):
        scores = np.zeros(self.arm_num)
        for i in range(scores.shape[0]):
            x[self.dim - self.arm_num + i] = 1.
            #print x
            #a = self.predict_proba(x)[1]
            #print a
            scores[i] = self.predict_proba(x)[1]
            x[self.dim - self.arm_num + i] = 0.
        return scores.argmax()

    def select_arm_conjugated(self,x):
        scores = np.zeros(self.arm_num)
        for i in range(self.arm_num):
            x[self.x_dim + i] = 1.
            for j in range(self.x_dim):
                x[self.x_dim + self.arm_num + j + self.x_dim * i] = x[j]
            scores[i] = self.predict_proba(x)[1]
            x[self.x_dim:] = 0.
        best_arm = scores.argmax()
        return best_arm

    def sample_data(self):
        x = np.zeros(self.dim)
        x[:self.x_dim] = np.random.normal(size = self.x_dim)
        return x
    
    def sample_label(self,x):
        proba = self.predict_proba(x,self.w_true)[1]
        return np.random.binomial(1,proba)

    def feature_embid(self,x,arm):
        x[self.x_dim + arm] = 1.
        for j in range(self.x_dim):
            x[self.x_dim + self.arm_num + j + self.x_dim * arm] = x[j]
        return x

class Egreedy_logistic(object):
    def __init__(self,x_dim,dim,arm_num,w,w_true=None,epsilon=0.1):
        self.x_dim = x_dim
        self.arm_num = arm_num
        self.dim = dim
        self.w = w
        self.epsilon = epsilon
        if w_true == None:
            self.w_true = np.random.normal(size=dim)
        else:
            self.w_true = w_true

    def predict_proba(self,X,w=None):
        if w == None:
            proba = 1 / (1 + np.exp(-1 * X.dot(self.w)))
            return np.array([1-proba , proba]).T
        else:
            proba = 1 / (1 + np.exp(-1 * X.dot(w)))
            return np.array([1-proba , proba]).T

    def select_arm(self,x):
        scores = np.zeros(self.arm_num)
        for i in range(self.arm_num):
            x[self.x_dim + i] = 1.
            for j in range(self.x_dim):
                x[self.x_dim + self.arm_num + j + self.x_dim * i] = x[j]
            scores[i] = self.predict_proba(x)[1]
            x[self.x_dim:] = 0.
        best_arm = scores.argmax()

        random_value = np.random.binomial(1,self.epsilon)
        if random_value == 0:
            return best_arm
        else:
            return np.random.randint(0,self.arm_num)

    def feature_embid(self,x,arm):
        x[self.x_dim + arm] = 1.
        for j in range(self.x_dim):
            x[self.x_dim + self.arm_num + j + self.x_dim * arm] = x[j]
        return x


start = time.time()
m = np.zeros(dim) ;q = np.ones(dim)*lam
T = 1000;n = 15
arm_num = 10
lam = 0.1; x_dim = 2; dim = x_dim + arm_num + arm_num*x_dim
arm = np.random.randint(0, arm_num)
w_true = np.random.normal(size=dim)#

epsilon = 0.03

ts_arms = np.zeros(T * n)
ts_results = np.zeros(T * n)

random_arms = np.zeros_like(ts_arms)
random_results = np.zeros_like(ts_results)

egreedy_arms = np.zeros_like(ts_arms)
egreedy_results = np.zeros_like(ts_results)

ts = Thompson_logistic(x_dim,dim,arm_num,lam,w_true)
lr_sgd_random = SGDClassifier(loss="log",alpha = lam , penalty = "l2")
lr_sgd_egreedy = SGDClassifier(loss="log",alpha = lam , penalty = "l2")
eg = Egreedy_logistic(x_dim,dim,arm_num,m,w_true,epsilon)

stack_random_X = np.zeros((1,dim))
stack_egreedy_X = np.zeros((1,dim))
stack_random_y = []
stack_egreedy_y = []

for t in range(T):
    ts_X = np.zeros((n,dim))
    ts_y = np.zeros(n)

    random_X = np.zeros_like(ts_X)
    random_y = np.zeros_like(ts_y)

    egreedy_X = np.zeros_like(ts_X)
    egreedy_y = np.zeros_like(ts_y)

    for j in range(n):
        #X[j],y[j] = ts.get_data_normal(1,arm)
        #X[j,:(dim-arm_num)] = np.random.normal(size = dim - arm_num)
        ts_X[j] = ts.sample_data()
        ts.sample_weight()
        ts_arm = ts.select_arm_conjugated(ts_X[j])

        random_arm = np.random.randint(0,arm_num)
        random_X[j] = ts.feature_embid(ts_X[j],random_arm) 

        egreedy_arm = eg.select_arm(ts_X[j])
        egreedy_X[j] = eg.feature_embid(ts_X[j],egreedy_arm)

        ts_X[j] = ts.feature_embid(ts_X[j],ts_arm)
        #X[j,dim-arm_num+arm] = 1. 
        ts_y[j] = ts.sample_label(ts_X[j])
        ts_arms[t*n + j] = ts_arm

        random_y[j] = ts.sample_label(random_X[j])
        random_arms[t*n + j] = random_arm

        egreedy_y[j] = ts.sample_label(egreedy_X[j])
        egreedy_arms[t*n + j] = egreedy_arm

    ts.update_with_grad(ts_X,ts_y)
    ts_results[t*n:(t+1)*n] = ts_y

    random_results[t*n:(t+1)*n] = random_y
    egreedy_results[t*n:(t+1)*n] = egreedy_y

    #
    stack_random_X = np.r_[stack_random_X,random_X]
    stack_random_y += random_y.tolist()
    if stack_random_y.count(0) != 0 and stack_random_y.count(1) != 0:
       lr_sgd_random.fit(np.array(stack_random_X)[1:],np.array(random_y) ,coef_init = lr_sgd_random.coef_)
       stack_random_X = np.zeros((1,dim))
       stack_random_y = []
    #random_results[t*n:(t+1)*n] = random_y

    #if np.any(np.bincount(egreedy_y.astype(np.int))==0):
    stack_egreedy_X = np.r_[stack_egreedy_X, egreedy_X]
    stack_egreedy_y += egreedy_y.tolist()
    if stack_egreedy_y.count(0) != 0 and stack_egreedy_y.count(1) != 0:
        lr_sgd_egreedy.fit(np.array(stack_egreedy_X)[1:] , np.array(stack_egreedy_y) , coef_init = lr_sgd_egreedy.coef_)
        eg.w = lr_sgd_egreedy.coef_[0]
        stack_egreedy_X = np.zeros((1,dim))
        stack_egreedy_y = []

print "ts_score:{}".format(ts_results[ts_results==1].shape[0]/float(ts_results.shape[0]))
print "random_score:{}".format(random_results[random_results==1].shape[0]/float(random_results.shape[0]))
print "egreedy_score:{}".format(egreedy_results[egreedy_results==1].shape[0]/float(egreedy_results.shape[0]))
print "time(sec):{}".format(time.time() - start)



# ---  experiment results (Thompson sampling seems to work better than other two methods)  ---

# T = 500, update ts by ts.update_with_grad() , which was fast
#ts_score:0.803333333333
#random_score:0.4988
#egreedy_score:0.6408
#time(sec):6.64178395271

# T = 500, update ts by ts.update_with_grad()
#ts_score:0.8516
#random_score:0.531333333333
#egreedy_score:0.7116
#time(sec):6.82829213142

# T = 500, update ts by ts.update() , which was slow
#ts_score:0.8032
#random_score:0.493733333333
#egreedy_score:0.678533333333
#time(sec):40.0501928329

# T = 1000, update ts by ts.update_with_grad() 
#ts_score:0.9182
#random_score:0.551
#egreedy_score:0.7818
#time(sec):13.2481338978

# ------------------
