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
    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = problem.compute_H(x)
    norm_g = np.linalg.norm(g, ord=inf)
    g0 = g
    # set initial iteration counter
    k = 0
    f_list = [f]
    #x_list = [x]
    
    while k < options.max_iterations and  norm_g >= options.term_tol*max(np.linalg.norm(g0, ord=inf), 1):
        if method.name == 'GradientDescent':
            x_new,f_new,g_new,d,alpha = algorithms.GDStep(x,f,g,problem,method,options)
            f_list.append(f_new) 

        elif method.name == 'Newton': 
            x_new,f_new,g_new,H_new,d,alpha = algorithms.NewtonStep(x,f,g,H,problem,method,options)
            H_old = H; H = H_new
            f_list.append(f_new) 
            '''
            if isinstance(f_new, list):
                f_list.append(f_new[0])
            else:
                f_list.append(f_new)           
            '''

        else:
            print('Warning: method is not implemented yet')
    
        # update old and new function values        
        x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord=inf)
        
        # increment iteration counter
        k = k + 1 

    return x,f, f_list
