import numpy as np
from .base import BaseOptimizer

class NewtonRaphson(BaseOptimizer):
    """
    Implementation of Newton-Raphson method.

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
        self.grad_ = self._grad_func(func)
        gfk = self.grad_(x0)
        self.hess_ = self._hess_func(func)
        hfk = self.hess_(x0)
        inv_hfk = np.linalg.inv(hfk)
        self.points_ = np.array([x0])

        self.num_steps_ = 0
        while self.num_steps_ <= self.MAX_DESC_ITER:
            pk = -np.dot(inv_hfk, gfk)
            alpha_opt = self._line_search(func, xk, pk)

            if alpha_opt is None or alpha_opt < self.eps:
                pk = -gfk
                alpha_opt = self._line_search(func, xk, pk)

            if alpha_opt is None or alpha_opt < self.eps:
                break

            xk_new = xk + pk*alpha_opt
            gfk_new = self.grad_(xk_new)
            self.points_ = np.append(self.points_, [xk_new], axis=0)
            self.num_steps_ += 1

            hfk = self.hess_(xk_new)
            inv_hfk = np.linalg.inv(hfk)

            if np.linalg.norm(gfk_new) < self.eps and np.linalg.norm(xk_new - xk) < self.eps:
                break

            xk = xk_new
            gfk = gfk_new
        return xk

    def _grad_func(self, func):
        return super()._grad_func(func)
    
    def _hess_func(self, func):
        return super()._hess_func(func)