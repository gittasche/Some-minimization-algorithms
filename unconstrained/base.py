from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from . import util
from matplotlib import cm

class BaseOptimizer(object):
    """
    Base class for all optimizers.
        
    Parameters
    ----------
    alpha0 : float
        initial vaulue for line search
    c1 : float
        Armijo condition parameter
    num_args : ndarray of int
        number of arguments
    """
    def __init__(self, alpha0=0.5, c1=0.1, num_args=0):
        self.alpha0 = alpha0
        self.c1 = c1
        self.num_args = num_args

    @abstractmethod
    def optimize(self, func=None, x0=None):
        raise NotImplementedError()

    def grad_func(self, func):
        """
        Automatic calculation of gradient
        analytic expression.

        Parameters
        ----------
        func : function
            function to minimize
        """
        return util.gradient(func, self.num_args)

    def hess_func(self, func):
        """
        Automatic calculation of hessian
        analytic expression.
        
        Parameters
        ----------
        func : function
            function to minimize
        """
        return util.hessian(func, self.num_args)
    
    def line_search(self, func, xk, pk, MAX_SRCH_ITER=100):
        """
        Backtracking Line Search.

        Parameters
        ----------
        func : function
            function to minimize
        xk : ndarray of float
            current point
        pk : ndarray of float
            current direction of search
        MAX_SRCH_ITER : int
            number of maximum iterations
        """
        first_cond = lambda alpha: func(*(xk + alpha*pk)) - func(*xk) - self.c1*alpha*np.dot(self.grad_(xk), pk)
        
        num_steps = 0
        alpha = self.alpha0
        while first_cond(alpha) > 0:
            alpha *= 0.5
            num_steps += 1
            if num_steps == MAX_SRCH_ITER:
                return None
        return alpha

    @staticmethod
    def plot(func, points, x_min, x_max, y_min, y_max, resolution=0.05):
        """
        Visualizing function and optimization path

        Parameters
        ----------
        func : function -> float
            function to minimize
        points : ndarray of float
            points of optimization path
        x_min, x_max, y_min, y_max : float
            borders of rectangular plot area
        resolution : float
            resolution of plot
        """
        X = np.arange(x_min, x_max, resolution)
        Y = np.arange(y_min, y_max, resolution)

        X, Y = np.meshgrid(X, Y)
        Z = func(X, Y)
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.5)
        ax.contour(X, Y, Z, zdir='Z', offset=-1, cmap=cm.jet, alpha=0.5)
        ax.plot(points[:,0], points[:,1], func(points[:,0], points[:,1]), '-o', color='black')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X, Y, Z, levels=50, cmap=cm.jet)
        ax.scatter(points[-1,0], points[-1,1], marker='o', s=50, color='red', zorder=1)
        ax.plot(points[:,0], points[:,1], '-x', zorder=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        plt.show()