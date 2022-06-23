import numpy as np
from .base import BaseOptimizer

class L_BFGS(BaseOptimizer):
    """
    Implementation of BFGS

    Parameters
    ----------
    M : int
        size of batch (prefer values 3 - 20)
    eps : float
        tolerance value for answer
    MAX_DESC_ITER : int
        number of maximum iterations
    alpha0 : float
        initial vaulue for line search
    c1 : float
        Armijo condition parameter
    """
    def __init__(self, M=10, eps=1e-7, MAX_DESC_ITER=2000, alpha0=0.5, c1=0.0001):
        self.M = M
        self.eps = eps
        self.MAX_DESC_ITER = MAX_DESC_ITER
        super().__init__(alpha0=alpha0, c1=c1)
        
    def optimize(self, func, x0):
        """
        Calculation of the minima.

        Parameters
        ----------
        func : function
            function to minimize
        x0 : ndarray of float
            initial point
        """
        xk = x0
        self.num_args = len(x0)
        self.grad_ = self._grad_func(func)
        gfk = self.grad_(xk)
        self.points_ = np.array([x0])
        
        k = 0
        s = np.array([x0])
        y = np.array([gfk])
        
        self.num_steps_ = 0
        while self.num_steps_ < self.MAX_DESC_ITER:
            inv_hfk = np.dot(s[k], y[k]) / np.dot(y[k], y[k]) * np.identity(self.num_args)
            
            pk = -self.BFGS_recursion(s, y, gfk, inv_hfk, k)
            alpha_opt = self.line_search(func, xk, pk)
            
            if alpha_opt is None:
                pk = -gfk
                alpha_opt = self.line_search(func, xk, pk)
            
            if alpha_opt is None:
                break
                
            xk_new = xk + alpha_opt*pk
            gfk_new = self.grad_(xk_new)
            
            if k >= self.M-1 or k == 0:
                s = np.delete(s, 0, 0)
                y = np.delete(y, 0, 0)
            else:
                k += 1
            
            s = np.append(s, [xk_new - xk], axis=0)
            y = np.append(y, [gfk_new - gfk], axis=0)
            
            if np.linalg.norm(gfk_new) < self.eps and np.linalg.norm(xk_new - xk) < self.eps:
                break
                
            xk = xk_new
            self.points_ = np.append(self.points_, [xk_new], axis=0)
            gfk = gfk_new
            self.num_steps_ += 1
        return self

    @staticmethod
    def BFGS_recursion(s, y, gfk, inv_hfk, k):
        """
        L-BFGS two-loop recursion algorithm.
        Algorithm 7.4 from Nocedal Wright book.

        Parameters
        ----------
        s : ndarray (m, dims) shape of float
            contains differences between points
        y : ndarray (m, dims) shape of float
            contains differences between gradients
        gfk : ndarray of float
            current gradient
        inv_hfk : ndarray of float
            inverse hessian
        k : int
            current number of items in batch
        """
        m = s.shape[0]
        alpha = np.zeros(m)
            
        q = gfk
        for i in range(k):
            alpha[m-i-1] = np.dot(s[m-i-1], q) / np.dot(s[m-i-1], y[m-i-1])
            q -= alpha[m-i-1] * y[m-i-1]
                
        r = np.dot(inv_hfk, q)
        for i in range(k):
            beta = np.dot(y[i], r) / np.dot(s[i], y[i])
            r += s[i] * (alpha[i] - beta)
        
        return r

    def _grad_func(self, func):
        return super().grad_func(func)