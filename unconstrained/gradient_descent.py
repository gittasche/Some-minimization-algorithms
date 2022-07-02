import numpy as np
from .base import BaseOptimizer

class GradientDescent(BaseOptimizer):
    """
    Implementation of gradient descent method.

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
    def __init__(self, eps=1e-7, MAX_DESC_ITER=2000, alpha0=0.5, c1=0.1):
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
        self.grad_ = self._grad_func(func)
        xk = x0
        pk = -self.grad_(x0)
        self.points_ = np.array([x0])
        
        self.num_steps_ = 0
        while self.num_steps_ < self.MAX_DESC_ITER:
            alpha_opt = self._line_search(func, xk, pk)

            if alpha_opt is None:
                break

            xk += alpha_opt*pk
            pk = -self.grad_(xk)
            self.num_steps_ += 1
            self.points_ = np.append(self.points_, [xk], axis=0)

            if np.linalg.norm(alpha_opt*pk) < self.eps:
                break
        return xk
    
    def _grad_func(self, func):
        return super()._grad_func(func)