import numpy as np
from .base import BaseOptimizer

class ConjugateGradients(BaseOptimizer):
    """
    Implementation of conjugate gradients method
    with two different formulas.

    Parameters
    ----------
    method : string
        "Fletcher-Reeves" or "Polak-Ribiere" methods of direction updating
    eps : float
        tolerance value for answer
    MAX_DESC_ITER : int
        number of maximum iterations
    alpha0 : float
        initial vaulue for line search
    c1 : float
        Armijo condition parameter
    """
    def __init__(self, method, eps=1e-7, MAX_DESC_ITER=2000, alpha0=0.5, c1=0.1):
        if method != 'Fletcher-Reeves' and method != 'Polak-Ribiere':
            raise RuntimeError('WRONG METOD: possible only Fletcher-Reeves or Polak-Ribiere')
        self.method = method
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
        self.num_args = len(x0)
        self.grad_ = self._grad_func(func)
        gfk = self.grad_(xk)
        pk = -gfk
        self.points_ = np.array([x0])
        
        self.num_steps_ = 0
        while self.num_steps_ < self.MAX_DESC_ITER:
            alpha_opt = self._line_search(func, xk, pk)
            
            if alpha_opt is None:
                pk = -gfk
                alpha_opt = self._line_search(func, xk, pk)
            
            if alpha_opt is None:
                break
            
            xk_new = xk + alpha_opt*pk
            gfk_new = self.grad_(xk_new)
            
            pk_new = -gfk_new + self._omega(gfk, gfk_new) * pk
            
            self.points_ = np.append(self.points_, [xk_new], axis=0)
            self.num_steps_ += 1
            
            if np.linalg.norm(pk_new) < self.eps and np.linalg.norm(xk_new - xk) < self.eps:
                break
            
            xk = xk_new
            gfk = gfk_new
            pk = pk_new
        return xk

    def _omega(self, gfk, gfk_new):
        if self.method == 'Fletcher-Reeves':
            omega = np.dot(gfk_new, gfk_new)/np.dot(gfk, gfk)
        elif self.method == 'Polak-Ribiere':
            omega = np.dot(gfk_new - gfk, gfk_new)/np.dot(gfk, gfk)
        return omega

    def _grad_func(self, func):
        return super()._grad_func(func)