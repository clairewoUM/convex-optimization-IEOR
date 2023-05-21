# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi & Seonho Woo & Vinayak Bassi


# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1) Rosenbrock function
# (2) Quadractic function
 
# Function that computes the function value for the Rosenbrock function
#
#           Input: x
#           Output: f(x)
#
import numpy as np

def rosen_func(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2


def rosen_grad(x):
    return np.array([-2*(1-x[0]) - 400*(x[1] - x[0]**2)*x[0], 200*(x[1] - x[0]**2)])


def rosen_Hess(x):
    return np.array([ [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200] ])  


def func2_func(x):
    term1 = (1.5 - x[0]*(1 - x[1]**1))**2
    term2 = (2.25 - x[0]*(1 - x[1]**2))**2
    term3 = (2.625 - x[0]*(1 - x[1]**3))**2
    return term1 + term2 + term3


def func2_grad(x):
    term1 = -2*(1-(x[1]**1))*(1.5 - x[0]*(1-x[1]**1))
    term2 = -2*(1-(x[1]**2))*(2.25 - x[0]*(1-x[1]**2))
    term3 = -2*(1-(x[1]**3))*(2.625 - x[0]*(1-x[1]**3))
    term4 = 2*(x[0])*(1.5 - x[0]*(1-x[1]**1))
    term5 = 2*(2*x[0]*x[1])*(2.25 - x[0]*(1-x[1]**2))
    term6 = 2*(3*x[0]*((x[1])**2))*(2.625 - x[0]*(1-x[1]**3))
    
    return np.array([term1+term2+term3, term4+term5+term6])

def func2_Hess(x):
    h11 = (2*(1-x[1])**2) + (2*(1-x[1]**2)**2) + (2*(1-x[1]**3)**2)
    h12 = (2*1*(x[1]**(1-1))*(1.5 - 2*x[0]*(1-x[1]**1))) + (2*2*(x[1]**(2-1))*(2.25 - 2*x[0]*(1-x[1]**2))) + (2*3*(x[1]**(3-1))*(2.625 - 2*x[0]*(1-x[1]**3)))
    h21 = (2*1*(x[1]**(1-1))*(1.5 - 2*x[0]*(1-x[1]**1))) + (2*2*(x[1]**(2-1))*(2.25 - 2*x[0]*(1-x[1]**2))) + (2*3*(x[1]**(3-1))*(2.625 - 2*x[0]*(1-x[1]**3)))
    h22 = (-2*(x[0]**2) + (4*2.25*x[0]) + 12*(x[0]**2)*(x[1]**2) + 12*2.625*x[0]*x[1] - 12*(x[0]**2)*x[1] + 30*(x[0]**2)*(x[1]**4))
    return np.array([[h11, h12], [h21, h22]])


def func3_func(x):
    f = ((np.exp(x[0])-1)/(np.exp(x[0])+1)) + 0.1*np.exp(-x[0])
    n = x.size
    
    for i in range(1, n):
        f = f + ((x[i]-1)**4)
    
    return f
    

def func3_grad(x):
    g = np.copy(x)
    g[0] = ((2*np.exp(x[0]))/((np.exp(x[0])+1)**2)) - 0.1*np.exp(-x[0])
    n = x.size
    
    for i in range(1, n):
        g[i] = 4*(x[i]-1)**3
    
    return g   
    

def func3_Hess(x):
    n = x.size
    H = np.eye(n)
    
    for i in range(1, n):
        H[i, i] = 12*(x[i]-1)**2
    
    H[0, 0] = ((2*np.exp(x[0])*(1-np.exp(x[0])))/((np.exp(x[0])+1)**3)) + 0.1*np.exp(-x[0])
    
    return H    
