import numpy as np
from .base import BaseOptimizer

class BFGS(BaseOptimizer):
    """
    Implementation of BFGS method.

    Parameters
    ----------
    eps : float
        tolerance value for answer
    MAX_DESC_ITER : int
        number of maximum iterations
    alpha0 : float
        initial vaulue for line search
    c1 : float
        Armijo condition parameter
    """
    def __init__(self, eps=1e-7, MAX_DESC_ITER=2000, alpha0=0.5, c1=0.0001):
        self.eps = eps
        self.MAX_DESC_ITER = MAX_DESC_ITER
        super().__init__(alpha0=alpha0, c1=c1)
        
    def optimize(self, func, x0):
        """
        Calculation of the minima.

        Parameters
        ----------
        func : function -> float
            function to minimize
        x0 : ndarray of float
            initial point
            
        Returns
        -------
        xk : ndarray of float
            final iteration point (it is a min point if algorighm converged)
        """
        xk = x0
        n = x0.shape[0]
        self.grad_ = self._grad_func(func)
        gfk = self.grad_(xk)
        inv_hfk = np.identity(n)
        self.points_ = np.array([x0])
        
        self.num_steps_ = 0
        while self.num_steps_ < self.MAX_DESC_ITER:
            pk = -np.dot(inv_hfk, gfk)
            alpha_opt = self._line_search(func, xk, pk)
            
            if alpha_opt is None:
                pk = -gfk
                alpha_opt = self._line_search(func, xk, pk)
                
            if alpha_opt is None:
                break
                
            sk = alpha_opt*pk
            xk_new = xk + sk
            gfk_new = self.grad_(xk_new)
            yk = gfk_new - gfk
            
            self.points_ = np.append(self.points_, [xk_new], axis=0)
            self.num_steps_ += 1
            
            inv_hfk = self._sherman_morrison(sk, yk, n, inv_hfk)
            if inv_hfk is None:
                break
            
            if np.linalg.norm(gfk_new) < self.eps and np.linalg.norm(xk_new - xk) < self.eps:
                break
                
            xk = xk_new
            gfk = gfk_new
        return xk
    
    @staticmethod
    def _sherman_morrison(sk, yk, dim, inv):
        """
        Sherman-Morrison formula to update
        inverse Hessian.

        Parameters
        ----------
        sk : ndarray of float
            difference between points
        yk : ndarray of float
            difference between gradients
        dim : int
            dimension of space
        inv : ndarray of float
            previos inverse matrix (ivnerse Hessian)
        """
        if np.dot(sk, yk) == 0:
            return None
        
        ro = 1./np.dot(sk, yk)
        A_left = np.identity(dim) - ro*np.outer(sk, yk.T)
        A_right = np.identity(dim) - ro*np.outer(yk, sk.T)
        inv = A_left @ inv @ A_right + ro*np.outer(sk, sk.T)

        return inv

    def _grad_func(self, func):
        return super()._grad_func(func)