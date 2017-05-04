# Bandits
implement famous bandit algorithms

## K-armed Bernoulli Bandits

* Epsilon-greedy
* UCB1
* KL-UCB
* Thompson sampling

Simple test result is: 

<img src="https://github.com/fullflu/bandit/blob/master/figures/regret_bernoulli.png">

```
python models.py
```


## Contextual Bandits (incomplete)

1: Thompson sampling action selection with online logistic regression , Algorithm 3 at Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).

2: Epsilon greedy action selection with online logistic regression

3: Random action selection with online logistic regression (for comparison)
