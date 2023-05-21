## Gradient Descent and Newton's Method
HW2 - Gradient Descent, Newton's Method, Backtracking Line Search

We will investigate the performance of Gradient Descent and Newton’s method on two unconstrained optimization problems.

### Algorithms:
* Gradient Descent with constant step size
* Gradient Descent with a backtracking line search
* Newton's Method with a backtracking line search
### Problems:
* Quadratic ($n=2$) Problem
$$f(x) = \frac{1}{2}x^TAx+b^Tx+c$$
where $x \in R^n$ ($n \in \{2, 10\}$) and $A$ is positive definite. The data ($A, b, x, x_0, x^{\*}$) for this problem is in the folder **Data**.
* Quadratic ($n=10$) Problem
* Rosenbrock Problem
<p align="center">
$f(x) = (1-w)^2 +100(z-w^2)^2$, where $x = [w$ $z]^T \in R^2$
</p>

#### Starting point:

$x_0 =\[1.2$ $1.2\]^T$. Note: $x^{\*} = \[w^{\*}$ $z^{\*}\]^T$ $=\[1$ $1\]^T$.

#### Constants:

$\bar{\alpha} =1$ ; $c_1 = 10^{-4}$; $\tau = 0.5$; max_iters = 100; $\epsilon = 10^{-6}$.

#### Termination Conditions:
<p align="center">
$\parallel \nabla f(x_k) \parallel_{inf} \leq \epsilon \mbox{max}$ { $\parallel \nabla f(x_0) \parallel_{inf}, 1$ }, or $k <$ max_iters ($k$: iteration counter).
<p>
