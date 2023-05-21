# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi & Seonho Woo

# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent

import numpy as np

def GDStep(x,f,g,problem,method,options):
    
    # Set the search direction d to be -g
    d = -g

    # determine step size
    if method.step_type == 'Constant':
        alpha = method.constant_step_size
        x_new = x + alpha*d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new) ### Add code (please replace None into its correct formula)
    elif method.step_type == 'Backtracking':
        tau = 0.5; c1 = 1e-4
        a0 = 1.0; alpha = a0
        f_a0 = problem.compute_f(x)
        f_new = problem.compute_f(x + a0*d)
        #print(f, f_a0)

        while not f_new <= (f_a0 + c1*alpha*g.T.dot(d)):
            alpha = alpha*tau
            f_new = problem.compute_f(x + alpha*d)
            #print(alpha, f_new)
        x_new = x + alpha*d # compute x_new with updated alpha
        g_new = problem.compute_g(x_new)
        #print(f_new)
    else:
        print('Warning: step type is not defined')

    return x_new,f_new,g_new,d,alpha


# 2) Newton's Method

def NewtonStep(x,f,g,H,problem,method,options):
    from numpy.linalg import inv

    d = -np.dot(np.linalg.inv(H), g)
    ##
    tau = 0.5; c1 = 1e-4
    a0 = 1.0; alpha = a0
    f_a0 = problem.compute_f(x)
    f_new = problem.compute_f(x + a0*d)
    #print(f, f_a0)

    while not f_new <= (f_a0 + c1*alpha*g.T.dot(d)):
        alpha = alpha*tau
        f_new = problem.compute_f(x + alpha*d)
        #print(alpha, f_new)
    x_new = x + alpha*d # compute x_new with updated alpha
    g_new = problem.compute_g(x_new)
    ##
    H_new = problem.compute_H(x_new)

    return x_new, f_new, g_new, H_new, d, alpha
