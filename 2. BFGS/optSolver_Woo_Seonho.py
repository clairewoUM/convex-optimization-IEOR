# IOE 511/MATH 562, University of Michigan
# Code written by: Seonho Woo

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
from math import inf
import numpy as np
import functions
import algorithms 

def optSolver_Woo_Seonho(problem,method,options):
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)  
    H = problem.compute_H(x)
    norm_g = np.linalg.norm(g, ord = np.inf)
    xk=[]
    fk=[]
    g0 = g
    # set initial iteration counter
    k = 0
    count = 0
    while  k < options.max_iterations and  norm_g >= options.term_tol*max(np.linalg.norm(g0, ord=np.inf), 1):
        
        if  (options.max_iterations<=k):
            return x,f,xk,fk, k, count
        elif (norm_g <= options.term_tol*max(np.linalg.norm(problem.compute_g(problem.x0), np.inf), 1)):
            return x, f, xk, fk, k, count    
        
        if method.name == 'GradientDescent':
            x_new,f_new,g_new, d, alpha = algorithms.gradient_descent_backtracking(f, g, x, problem, method, options, alpha=1, tau=0.5, c1= 1e-4, max_iter=1000, eps=1e-6)
        elif method.name == 'Newton': 
            x_new,f_new,g_new, d, alpha, count = algorithms.modified_newtons_method(f, g, H, x, problem, method, options, max_iter=1000, tol=1e-6)
        elif method.name == 'BFGS': 
            x_new,f_new,g_new, d, alpha, count = algorithms.bfgs(f, g, x, problem, method, options, eps=1e-6, max_iter=1000)
        elif method.name == 'L-BFGS_2': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=2, k = k, max_iter=1000, epsilon=1e-6)
        elif method.name == 'L-BFGS_5': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=5, k = k,  max_iter=1000, epsilon=1e-6)
        elif method.name == 'L-BFGS_10': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=10, k = k,  max_iter=1000, epsilon=1e-6)
    
        else:
            print('Warning: method is not implemented yet')
    
        # update old and new function values        
        if k==0:
            xk.append(x)
            fk.append(f)
        x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g, ord= np.inf)

        xk.append(x); fk.append(f)
        # increment iteration counter
        k +=  1 
        
    return x,f,xk,fk, k, count
