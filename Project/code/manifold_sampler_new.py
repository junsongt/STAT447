import numpy as np
from scipy import stats
import scipy.linalg as la
import math


class Manifold_sampler_new:

    def __init__(self, x, q, G, H, constraints, f=1, eps=1e-8, nmax=10):
        self.x = x
        self.q = q
        self.G = G
        self.constraints = constraints
        self.f = f
        self.eps = eps
        self.nmax = nmax
        self.G_x = G(x)
        self.H = H
        self.L_x = la.cholesky(self.G_x.T @ self.G_x, lower=True)
        self.H_x = self.H(x)
        self.A_x = la.cholesky(self.H_x, lower=True)

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

        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            print("Invalid point encountered in Newton solver:", v)
            return {"pt": v, "flag": False}  # Reject invalid point immediately

        # Dimension of the problem
        d = G_x.shape[1]
        a = np.zeros(d)
        y = v + np.dot(G_x, a)
        qval = self.q(y)
        qerr = la.norm(qval)

        i = 1
        found = False
        while i <= self.nmax and qerr > self.eps:
            delta_a = la.solve_triangular(
                L_x.T, la.solve_triangular(L_x, -qval, lower=True), lower=False
            )
            a += delta_a
            y = v + np.dot(G_x, a)
            qval = self.q(y)
            qerr = la.norm(qval)
            i += 1

        if i <= self.nmax:
            found = True

        return {"pt": y, "flag": found}

    def sample(self, x):
        """
        Sampling method for the manifold.
        """
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Invalid point encountered:", x)
            return x  # Reject invalid points early

        d = len(x)
        z = np.random.normal(0, 1, d)
        cov = np.linalg.inv(self.H_x)
        zeta = 0.1* la.cholesky(cov, lower=True) @ z
        # zeta = la.solve_triangular(self.A_x.T, z, lower=False)
        xi = la.solve_triangular(
            self.L_x.T,
            la.solve_triangular(self.L_x, self.G_x.T @ zeta, lower=True),
            lower=False,
        )
        v_x = zeta - self.G_x @ xi

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
        H_y = self.H(y)
        p_vx = math.exp(-0.5 * v_x.T @ self.H_x @ v_x)
        p_vy = math.exp(-0.5 * v_y.T @ H_y @ v_y)
        # p_vx = np.exp(-np.linalg.norm(v_x) ** 2 / (2 * self.s**2))
        # p_vy = np.exp(-np.linalg.norm(v_y) ** 2 / (2 * self.s**2))
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

        A_y = la.cholesky(H_y, lower=True)
        self.A_x = A_y

        return y
