# Bandits
implement famous bandit algorithms

## Requirements

I checked algorithms in the following environment:

* OS X (10.9+), CPU
* python (2.7.11)
* numpy (1.11.3)
* matplotlib (1.4.3)

## K-armed Bernoulli Bandits

* Epsilon-greedy
* UCB
* KL-UCB
* Thompson sampling

Simple test can be done as follows:
```
python models.py
```

The result is shown as follows: 

<img src="https://github.com/fullflu/bandit/blob/master/figures/regret_bernoulli.png">

## Contextual Bandits (incomplete)

1: Thompson sampling action selection with online logistic regression , Algorithm 3 at Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).

2: Epsilon greedy action selection with online logistic regression

3: Random action selection with online logistic regression (for comparison)



