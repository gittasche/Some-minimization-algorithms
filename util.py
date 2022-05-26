import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def grad(func, x0, step):
    '''
    Numerical gradient calculation
    '''
    n = len(x0)
    gradient = np.zeros(n)

    for i in range(n):
        x_new = x0.copy()
        x_new[i] += step
        gradient[i] = func(*x_new) - func(*x0)
        
    return gradient/step

def line_search(func, xk, pk, step, alpha0, c1, MAX_SRCH_ITER=100):
    '''
    Backtracking Line Search
    '''
    first_cond = lambda alpha: func(*(xk + alpha*pk)) - func(*xk) - c1*alpha*np.dot(grad(func, xk, step), pk)
    
    num_steps = 0
    alpha = alpha0
    while first_cond(alpha) > 0:
        alpha *= 0.5
        num_steps += 1
        if num_steps == MAX_SRCH_ITER:
            return None
    return alpha

def hessian_calc(func, x0, step):
    '''
    Numerical hessian calculation for Newton-Raphson method
    '''
    n = len(x0)
    hessian = np.zeros([n,n])
    
    for i in range(n):
        x_new = np.copy(x0)
        x_new[i] += step
        hessian[i] = grad(func, x_new, step) - grad(func, x0, step)
        
    return hessian/step