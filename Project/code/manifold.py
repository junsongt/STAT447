import numpy as np
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import jax
from jax import random
import jax.numpy as jnp
from jax import grad, value_and_grad
import math

from sympy import false, true


# ===============================================================================
def newton_solve(v, q, G_x, L_x, nmax, eps):
    # guard inf case
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        print("Invalid point encountered in Newton solver:", v)
        return {"pt": v, "flag": False}  # Reject invalid point immediately
    # Dimension of the problem
    d = G_x.shape[1]
    a = np.zeros(d)
    y = v + np.dot(G_x, a)
    qval = q(y)
    qerr = la.norm(qval)
    # guard inf case
    if np.any(np.isnan(qval)) or np.any(np.isinf(qval)):
        print("Invalid qval at start of Newton solver:", qval)
        return {"pt": v, "flag": False}  # Reject invalid point
    i = 1
    found = False
    while i <= nmax and qerr > eps:
        delta_a = la.solve_triangular(
            L_x.T, la.solve_triangular(L_x, -qval, lower=True), lower=False
        )
        a += delta_a
        y = v + np.dot(G_x, a)
        qval = q(y)
        # guard inf case
        if np.any(np.isnan(qval)) or np.any(np.isinf(qval)):
            print("Invalid qval at start of Newton solver:", qval)
            return {"pt": v, "flag": False}  # Reject invalid point

        qerr = la.norm(qval)
        i += 1
    if i <= nmax:
        found = True
    return {"pt": y, "flag": found}


# ===================================================================================
def sample(x, q, f, G, G_x, L_x, s, nmax, eps):
    # guard inf case
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print("Invalid point encountered:", x)
        return {"pt": x, "grad": G_x, "chol": L_x}  # Reject invalid points early

    d = len(x)
    z = np.random.normal(0, 1, d)
    xi = la.solve_triangular(
        L_x.T,
        la.solve_triangular(L_x, G_x.T @ z, lower=True),
        lower=False,
    )
    v_x = z - G_x @ xi

    # guard inf case
    if np.any(np.isnan(v_x)) or np.any(np.isinf(v_x)):
        print("Invalid tangent vector v_x:", v_x)
        return {
            "pt": x,
            "grad": G_x,
            "chol": L_x,
        }  # Reject the step and return the current point

    proj_for = newton_solve(x + s * v_x, q, G_x, L_x, nmax, eps)
    if not proj_for["flag"]:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Projection failed

    y = proj_for["pt"]
    # for h in constraints:
    #     if h(y) < 0:
    #         return x  # Constraint violated

    G_y = G(y)
    L_y = la.cholesky(G_y.T @ G_y, lower=True)
    delta_xy = x - y
    xi = la.solve_triangular(
        L_y.T, la.solve_triangular(L_y, G_y.T @ delta_xy, lower=True), lower=False
    )
    v_y = (delta_xy - G_y @ xi) / s
    # p_vx = math.exp(-np.linalg.norm(v_x) ** 2 / (2 * s**2))
    # p_vy = math.exp(-np.linalg.norm(v_y) ** 2 / (2 * s**2))
    # alpha = min(1, (f(y) * p_vy) / (f(x) * p_vx))
    alpha = min(
        1,
        math.exp(
            math.log(f(y))
            - math.log(f(x))
            - (np.linalg.norm(v_y) ** 2 - np.linalg.norm(v_x) ** 2) / 2
        ),
    )
    u = np.random.uniform(0, 1)
    if u > alpha:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Rejection

    proj_rev = newton_solve(y + s * v_y, q, G_y, L_y, nmax, eps)
    if not proj_rev["flag"]:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Reverse projection failed

    xx = proj_rev["pt"]
    if la.norm(xx - x) > eps:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Reject due to distance criterion

    G_x = G_y
    L_x = L_y

    return {"pt": y, "grad": G_y, "chol": L_y}
