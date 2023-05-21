# SG: Stochastic Gradient
HW4 - Stochastic Gradient Method, Gradient Descent Method

We will investigate the performance of the Stochastic Gradient method and Gradient Descent on binary classification tasks that arise in Machine Learning

### Algorithms:
* (GD) Gradient Descent (with a backtracking line search)
* (SG) Stochastic Gradient (with a fixed steplength, $\alpha_k = \alpha$
* (SG) Stochastic Gradient (with a diminishing steplength, $\alpha_k = \frac{\alpha}{k}$

For the SG variants: (1) mini-batch $b=1$, unless otherwise specified, (2) sample data points uniformly at random with replacement.

**Data**: The data description below:

1. *autralian*: Credit card applications dataset. Goal is to classify good/bad credit card applicants.

(Details: $n = n_{train}+ n_{test} = 621, n_{train} = 435, n_{test} = 186, d=14$)

2. *mushroom*: Dataset with information about mushrooms. Goal is to classify edible/inedible (poisonous) mushrooms.

(Details: $n = n_{train}+ n_{test} = 5500, n_{train} = 3850, n_{test} = 1650, d=112$)

### Loss Functions:
$w \in R^d$ (weights), $X \in R^{n\times d}$ (data matrix), $X_i \in R^{1\times d}(i$ th row of data matrix), $y \in R^n$ (labels), $y_i \in \{-1, 1\}$ $(i$ th label),

1. Linear Least Squares:

$$F(w) = \frac{1}{2n} \sum_{i=1}^n ||X_iw-y_i||_2^2.$$

2. Logistic Regression:

$$F(w) = \frac{1}{n} \sum_{i=1}^n log(1+e^{-y_i(X_iw)}).$$

Starting Points:
$w_0 = 0 \in R^d$

### Termination Conditions:
We will stop algorithms after $20n$ gradient evaluations have been computed.
