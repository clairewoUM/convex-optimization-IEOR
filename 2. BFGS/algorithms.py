# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi & Seonho Woo

# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent

import numpy as np

def gradient_descent_backtracking(f, g, x, problem,method,options, alpha=1, tau=0.5, c1=1e-4, max_iter=1000, eps=1e-6):
    """
    Gradient descent algorithm with backtracking line search
    :param f: function to optimize
    :param grad: gradient of the function
    :param x_init: initial guess for the optimization
    :param alpha_init: initial step size for the line search
    :param rho: parameter for the backtracking line search
    :param c: parameter for the backtracking line search
    :param max_iter: maximum number of iterations
    :param tol: tolerance for the stopping criterion
    :return: optimal solution x_opt and the function value f(x_opt)
    """
    if method.step_type == 'Backtracking':
        x = x
        alpha = alpha
        grad_x = problem.compute_g(x)
        while problem.compute_f(x - alpha * grad_x) > problem.compute_f(x) - c1 * alpha * np.dot(grad_x, grad_x):
            alpha = tau * alpha
        # update x
        x = x - alpha * grad_x
        
        if np.linalg.norm(grad_x) < eps:
            return x, problem.compute_f(x), problem.compute_g(x), -problem.compute_g(x), alpha
    else:
        print('Warning: step type is not defined')
    return x, problem.compute_f(x), problem.compute_g(x), -problem.compute_g(x), alpha

#####################################################################################################################################################################

# Newton's method with modifications

from numpy.linalg import LinAlgError, cholesky, solve


def modified_newtons_method(f, g, H, x, problem, method, options,  max_iter=1000, tol=1e-6):
    """
    Modified Newton's method with backtracking line search and Cholesky factorization for the Hessian.

    Parameters:
    x (numpy.ndarray): initial point
    max_iter (int): maximum number of iterations (default: 1000)
    tol (float): tolerance for stopping criterion (default: 1e-6)

    Returns:
    numpy.ndarray: optimal point
    """
    if method.step_type == 'Modified':
        modified = 0
        g = problem.compute_g(x)
        H = problem.compute_H(x)
        try:
            L = cholesky(H)
            d = -solve(H, g)
        except LinAlgError:
            d = -np.linalg.solve(np.eye(len(x)), g)
        alpha = 1.0
        while problem.compute_f(x + alpha * d) > problem.compute_f(x) + 1e-4 * alpha * g @ d:
            alpha *= 0.5
            modified += 1
        x_new = x + alpha * d
        if np.linalg.norm(x_new - x) < tol:
            return x_new, problem.compute_f(x_new), problem.compute_g(x_new), d, alpha, modified
        x = x_new
    else:
        print('Warning: step type is not defined') 
        
    return x, problem.compute_f(x), problem.compute_g(x), d, alpha, modified

#####################################################################################################################################################################
# BFGS method

def bfgs(f, g, x0,problem,method,options,  eps=1e-6, max_iter=1000):

    if method.step_type == 'Backtracking':
        skipcount = 0
        n = len(x0)
        Hk = np.eye(n)
        xk = x0
        fk = problem.compute_f(xk)
        g = problem.compute_g(xk)

        desc_dir = -np.linalg.solve(Hk, g)
        if np.linalg.norm(desc_dir) < eps:
            pass
        alpha = backtracking_line_search(f, g, xk, desc_dir,problem,method,options)
        xk_new = xk + alpha * desc_dir
        sk = xk_new - xk
        yk = problem.compute_g(xk_new) - g
        if np.dot(sk, yk) < eps * np.linalg.norm(sk, 2)*np.linalg.norm(yk, 2):
            skipcount += 1
        rho = 1 / np.dot(yk, sk)
        if np.isinf(rho):
            return xk, fk, g, desc_dir, alpha, skipcount
        A = np.eye(n) - rho * np.outer(sk, yk)
        B = np.eye(n) - rho * np.outer(yk, sk)

        Hk = np.dot(A, np.dot(Hk, B)) + rho * np.outer(sk, sk)
        xk = xk_new
        fk = problem.compute_f(xk)
        g = problem.compute_g(xk)
    else:
        print('Warning: step type is not defined')
        
    return xk, fk, g, desc_dir, alpha, skipcount
#####################################################################################################################################################################
# Backtracking line search function

def backtracking_line_search(f, g, x, desc_dir,problem,method,options, alpha=1, tau=0.9, c1=1e-4):
    if method.step_type == 'Backtracking':
        fk = problem.compute_f(x)
        g = problem.compute_g(x)
        while problem.compute_f(x + alpha * desc_dir) > fk + c1 * alpha * np.dot(g, desc_dir):
            alpha *= tau
    else:
        print('Warning: step type is not defined')
    return alpha

#####################################################################################################################################################################

def L_BFGS(f, g, x0, problem,method,options, m, k, max_iter=1000, epsilon=1e-6):
    """
    L-BFGS optimization algorithm with memory, initial Hessian approximation as identity, and skip update.

    Parameters:
    -----------
    fun : function
        The objective function to be minimized.
    grad : function
        The gradient function of the objective function.
    x0 : array_like
        The initial guess for the optimizer.
    max_iter : int, optional
        The maximum number of iterations to run the optimizer. Default is 100.
    m : int, optional
        The number of previous iterations to store in memory. Default is 10.
    epsilon : float, optional
        The tolerance for convergence. Default is 1e-5.

    Returns:
    --------
    x : array_like
        The optimal point found by the optimizer.
    fval : float
        The value of the objective function at the optimal point.
    """

    if method.step_type == 'Two-loop':
        skipcount = 0
        n = len(x0)
        x = x0.copy()
        fx = problem.compute_f(x)
        gx = problem.compute_g(x)
        Hk = np.eye(n)
        s = []
        y = []
        rho = []

        pk = -np.dot(Hk, gx)

        # backtracking line search
        alpha = 1.0
        c1 = 1e-4
        tau = 0.5
        while problem.compute_f(x + alpha*pk) > fx + c1*alpha*np.dot(gx, pk):
            alpha *= tau
        xp = x.copy()
        x += alpha*pk
        fp = fx
        fx = problem.compute_f(x)
        gp = gx
        gx = problem.compute_g(x)

        sk = x - xp
        yk = gx - gp
        rhok = 1.0 / np.dot(yk, sk)

        if np.dot(yk, sk) < epsilon*np.linalg.norm(sk, 2)*np.linalg.norm(yk, 2):
            skipcount += 1           

        if len(s) is not 0:
            if np.dot(yk, sk) < epsilon*np.linalg.norm(sk, 2)*np.linalg.norm(yk, 2):
                skipcount += 1  

            if k < m:
                s.append(sk)
                y.append(yk)
                rho.append(rhok)
            else:
                s.pop(0)
                y.pop(0)
                rho.pop(0)
                s.append(sk)
                y.append(yk)
                rho.append(rhok) 

            q = gx.copy()
            alpha_list = []
            
            #for i in range(k-1, k-m, -1):
            for i in range(len(s)-1, -1, -1):
                alpha_i = rho[i] * np.dot(s[i], q)
                alpha_list.append(alpha_i)
                q -= alpha_i * y[i]
            r = Hk.dot(q)
            
            #for i in range(k-1, k-m, -1):
            for i in range(len(s)):
                beta = rho[i] * np.dot(y[i], r)
                r += s[i] * (alpha_list[-i-1] - beta)
            
            pk = -r
    
        if np.linalg.norm(gx) < epsilon:
            return x, fx, gx, r, skipcount

        if np.dot(gx, sk) > 0:
            Hk = np.eye(n)
    else:
        print('Warning: step type is not defined')    
        
    return x, fx, gx, skipcount


def lbfgs(f, g, x0, problem,method,options, m, k, max_iter=1000, epsilon=1e-6):#(x0, fun, grad, args=(), m=10, epsilon=1e-5, max_iter=1000):
    """
    Minimize a function using the L-BFGS algorithm.

    Parameters
    ----------
    x0 : array_like
        Initial guess
    m : int, optional
        Number of corrections to approximate the inverse Hessian.
    epsilon : float, optional
        Threshold for skipping update.

    Returns
    -------
    x : ndarray
        Minimum point.
    f : float
        Minimum value.
    k : int
        Number of iterations.
    g : ndarray
        Gradient at the minimum point.
    """
    if method.step_type == 'Two-loop':
        x = np.asarray(x0).flatten()
        n = len(x)
        s_list = []
        y_list = []
        rho_list = []
        alpha_list = []
        count_skip = 0

        f, g = problem.compute_f(x), problem.compute_g(x)
        if np.linalg.norm(g) < epsilon:
            return x, f, g, count_skip
        
        p = -g
        if k == 0:
            H0 = np.eye(n)
            y = g / np.linalg.norm(g)
            s = H0.dot(p)
        else:
            i = k % m
            alpha = alpha_list[-1-i]
            s = s_list[-1-i]
            y = y_list[-1-i]
            rho = rho_list[-1-i]
            alpha_list[-1-i] = rho * np.dot(s, p) / np.dot(y, p)
            p -= alpha_list[-1-i] * y
            s = H0.dot(p)
            for j in range(min(k, m)):
                i = (k-j-1) % m
                alpha = alpha_list[-1-i]
                s_j = s_list[-1-i]
                y_j = y_list[-1-i]
                rho_j = rho_list[-1-i]
                beta = rho_j * np.dot(y_j, s) / np.dot(y_j, y_j)
                s -= beta * s_j
                alpha_list[-1-i] = rho_j * np.dot(s_j, s) / np.dot(y_j, s)
                s -= alpha_list[-1-i] * y_j

        if np.dot(s, y) < epsilon * np.linalg.norm(s, 2) * np.linalg.norm(y, 2):
            count_skip += 1
            #continue

        alpha = 1.0
        c1 = 1e-4
        tau = 0.5
        while problem.compute_f(x + alpha * p) > f + c1 * alpha * np.dot(g, p):
            alpha *= tau
        p *= alpha
        x = x + p
        s_list.append(s)
        y_list.append(g - problem.compute_g(x))
        rho_list.append(1 / np.dot(y_list[-1], s_list[-1]))
        alpha_list.append(alpha)

    else:
        print('Warning: step type is not defined')    

    return x, f, g, count_skip
    
'''
def LBFGS_newnew(f, g, x, problem, method, options, m, k, max_iter=1000, epsilon=1e-6):

    #step_type = method['options']['step_type']
    x_new = x
    gx = problem.compute_g(x)
    gp = gx
    sk = x_new - x
    yk = gx - gp
    rhok = 1.0 / np.dot(yk, sk)
    size_k = len(x)
    H = np.eye(size_k)    #H = np.eye(size_k)

    if method.step_type == 'Two-loop':

        if size_k != 0:
            # two loop recursion to update H
            q = g
            alpha = []
            for i in range(size_k-1, -1, -1):
                rho_i = 1 / (y_k[:, i] @ s_k[:, i])
                alpha_i = rho_i * (s_k[:, i] @ q)
                alpha.insert(0, alpha_i)
                q = q - alpha_i * y_k[:, i]

            r = H @ q
            for i in range(size_k):
                beta = 1 / (y_k[:, i] @ s_k[:, i])
                beta *= y_k[:, i] @ r
                r = r + s_k[:, i] * (alpha[i] - beta)
            d = -r
        else:
            # search direction is -Hxg where H is inverse hessian
            d = -H @ g

        # Stepsize calculation using Backtracking
        alpha = 1.0
        c1 = 1e-4
        x_new = x.copy()
        x_new.x0 = x.x0 + alpha * d
        f_new = problem.compute_f(x_new)

        # Satisfying Armijo
        while f_new > f + c1 * alpha * (g @ d):
            alpha *= 0.5
            x_new.x0 = x.x0 + alpha * d
            f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

        # updates for s_k & y_k
        s_new = x_new.x0 - x.x0
        y_new = g_new - g
        if s_new @ y_new > options.term_tol * np.linalg.norm(s_new, 2) * np.linalg.norm(y_new, 2) and size_k == m:
            # update s_k and y_k
            s_k = np.hstack([s_k[:, 1:], s_new.reshape((-1, 1))])
            y_k = np.hstack([y_k[:, 1:], y_new.reshape((-1, 1))])
        elif s_new @ y_new > options.term_tol * np.linalg.norm(s_new, 2) * np.linalg.norm(y_new, 2) and size_k < m:
            s_k = np.hstack([s_k, s_new.reshape((-1, 1))])
            y_k = np.hstack([y_k, y_new.reshape((-1, 1))])
        else:
            #disp('No Update i.e. skipped');
            pass

    return x_new, f_new, g_new, d, alpha, s_k, y_k
'''
