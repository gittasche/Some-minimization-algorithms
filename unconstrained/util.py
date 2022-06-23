import autograd.numpy as np
from autograd import grad

# Some utility math functions

def gradient(func, num_args):
    """
    Return gradient vector function in convinient form

    Parameters
    ----------
    func : function -> float
        current function
    num_args : int
        number of function arguments
    """
    grad_expr = [grad(func, i) for i in range(num_args)]
    def grad_calc(x):
        return np.array([grad_expr[i](*x) for i in range(num_args)])
    return grad_calc

def hessian(func, num_args):
    """
    Return hessian matrix function in convinient form

    Parameters
    ----------
    func : function -> float
        current function
    num_args : int
        number of function arguments
    """
    hess_expr = [[grad(grad(func, j), i) for j in range(num_args)] for i in range(num_args)]
    def hess_calc(x):
        return np.array([[hess_expr[i][j](*x) for j in range(num_args)] for i in range(num_args)])
    return hess_calc