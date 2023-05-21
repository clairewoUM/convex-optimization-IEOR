# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi & Seonho Woo


# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1) Rosenbrock function
# (2) Quadractic function
 
# Function that computes the function value for the Rosenbrock function
#
#           Input: x
#           Output: f(x)
#
import numpy as np
import scipy.io as sio

def rosen_func(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

# Function that computes the gradient of the Rosenbrock function
# 
#           Input: x
#           Output: g = nabla f(x)
# 

def rosen_grad(x):
    return np.array([
        2*(x[0]-1) - 4*100*x[0]*(x[1]-x[0]**2), 2*100*(x[1]-x[0]**2)
    ])

# Function that computes the Hessian of the Rosenbrock function
#
#           Input: x
#           Output: H = nabla^2 f(x)
#

def rosen_Hess(x):
    return np.matrix([
        [2 - 4*100*(x[1] - 3*x[0]**2), -4*100*x[0]], [-4*100*x[0], 2*100]
    ])


# Function that computes the function value for the Quadractic function
#
#           Input: x
#           Output: f(x)
#

def quad_func(x):
    filepath = r'/Users/clairewoo/Desktop/IOE 511/Data/'
    data2 = sio.loadmat(filepath + "quadratic2.mat")
    data10 = sio.loadmat(filepath + "quadratic10.mat")

    if np.shape(x)[0] == 2:
        A = data2['A']
        b = data2['b']
        c = data2['c']

    elif np.shape(x)[0] == 10: 
        A = data10['A']
        b = data10['b']
        c = data10['c']

    return 1/2*np.dot(np.dot(x.T, A), x) + np.dot(b.T, x) + c

def quad_grad(x):
    filepath = r'/Users/clairewoo/Desktop/IOE 511/Data/'
    data2 = sio.loadmat(filepath + "quadratic2.mat")
    data10 = sio.loadmat(filepath + "quadratic10.mat")

    if np.shape(x)[0] == 2:
        A = data2['A']
        b = data2['b']
        c = data2['c']

    elif np.shape(x)[0] == 10: 
        A = data10['A']
        b = data10['b']
        c = data10['c']

    return 1/2*(A+A.T).dot(x) + b

def quad_hess(x):
    filepath = r'/Users/clairewoo/Desktop/IOE 511/Data/'
    data2 = sio.loadmat(filepath + "quadratic2.mat")
    data10 = sio.loadmat(filepath + "quadratic10.mat")
    
    if np.shape(x)[0] == 2:
        A = data2['A']
        b = data2['b']
        c = data2['c']

    elif np.shape(x)[0] == 10: 
        A = data10['A']
        b = data10['b']
        c = data10['c']

    return 1/2*(A+A.T)
