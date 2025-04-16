import numpy as np
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt
import jax
from jax import random
import jax.numpy as jnp
from jax import grad, value_and_grad
import math


class Manifold_sampler:

    def __init__(self, x, q, G, constraints, s, f=1, eps=1e-8, nmax=10):
        self.x = x
        self.q = q
        self.G = G
        self.constraints = constraints
        self.s = s
        self.f = f
        self.eps = eps
        self.nmax = nmax
        self.G_x = G(x)
        self.L_x = la.cholesky(self.G_x.T @ self.G_x, lower=True)

    def newton_solve(self, v, G_x, L_x):
        """
        Newton's method solver.
        Parameters:
            v: numpy array, initial guess vector.
            G_x: numpy array, gradient matrix.
            R_x: numpy array, triangular matrix (e.g., Cholesky factorization).
            q: callable, function that maps a vector to its residual.
            eps: float, tolerance for convergence.
            nmax: int, maximum number of iterations.
        Returns:
            a dictionary with point: y; flag: found.
        """

        # guard inf case
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            print("Invalid point encountered in Newton solver:", v)
            return {"pt": v, "flag": False}  # Reject invalid point immediately

        # Dimension of the problem
        d = G_x.shape[1]
        a = np.zeros(d)
        y = v + np.dot(G_x, a)
        qval = self.q(y)
        qerr = la.norm(qval)
        # guard inf case
        if np.any(np.isnan(qval)) or np.any(np.isinf(qval)):
            print("Invalid qval at start of Newton solver:", qval)
            return {"pt": v, "flag": False}  # Reject invalid point

        i = 1
        found = False
        while i <= self.nmax and qerr > self.eps:
            delta_a = la.solve_triangular(
                L_x.T, la.solve_triangular(L_x, -qval, lower=True), lower=False
            )
            a += delta_a
            y = v + np.dot(G_x, a)
            qval = self.q(y)

            # guard inf case
            if np.any(np.isnan(qval)) or np.any(np.isinf(qval)):
                print("Invalid qval at start of Newton solver:", qval)
                return {"pt": v, "flag": False}  # Reject invalid point
        
            qerr = la.norm(qval)
            i += 1

        if i <= self.nmax:
            found = True

        return {"pt": y, "flag": found}

    def sample(self, x):
        """
        Python translation of the manifold_sampler function.

        Parameters:
            x (numpy array): Initial point on the manifold.
            G_x (numpy array): Gradient matrix at x.
            R_x (numpy array): Cholesky factor of G_x.T @ G_x.
            constraints (list): List of constraint functions.
            G (callable): Function to compute gradient matrix at a given point.
            f (callable): Target density function.
            newton_solver (callable): Function for Newton projection.
            s (float): Scaling factor.
            eps (float): Tolerance for rejection criteria.

        Returns:
            numpy array: The new sampled point on the manifold.
        """
        # guard inf case
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Invalid point encountered:", x)
            return x  # Reject invalid points early

        d = len(x)
        z = np.random.normal(0, 1, d)
        xi = la.solve_triangular(
            self.L_x.T,
            la.solve_triangular(self.L_x, self.G_x.T @ z, lower=True),
            lower=False,
        )
        v_x = self.s * (z - self.G_x @ xi)

        # guard inf case
        if np.any(np.isnan(v_x)) or np.any(np.isinf(v_x)):
            print("Invalid tangent vector v_x:", v_x)
            return x  # Reject the step and return the current point


        proj_for = self.newton_solve(x + v_x, self.G_x, self.L_x)
        if not proj_for["flag"]:
            return x  # Projection failed

        y = proj_for["pt"]
        for h in self.constraints:
            if h(y) < 0:
                return x  # Constraint violated

        G_y = self.G(y)
        L_y = la.cholesky(G_y.T @ G_y, lower=True)
        delta_xy = x - y
        xi = la.solve_triangular(
            L_y.T, la.solve_triangular(L_y, G_y.T @ delta_xy, lower=True), lower=False
        )
        v_y = delta_xy - G_y @ xi
        p_vx = math.exp(-np.linalg.norm(v_x) ** 2 / (2 * self.s**2))
        p_vy = math.exp(-np.linalg.norm(v_y) ** 2 / (2 * self.s**2))
        alpha = min(1, (self.f(y) * p_vy) / (self.f(x) * p_vx))
        u = np.random.uniform(0, 1)
        if u > alpha:
            return x  # Rejection

        proj_rev = self.newton_solve(y + v_y, G_y, L_y)
        if not proj_rev["flag"]:
            return x  # Reverse projection failed

        xx = proj_rev["pt"]
        if la.norm(xx - x) > self.eps:
            return x  # Reject due to distance criterion

        self.G_x = G_y
        self.L_x = L_y

        return y
