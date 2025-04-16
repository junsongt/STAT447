import numpy as np
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
from jax import random
import jax.numpy as jnp
from jax import grad, value_and_grad
import math

# ==================================================================
# cone global
s = 0.05
n_iter = 10000  # number of samples
# constraints are {h_i(x) : h_i(x) >= 0}
constraints = [lambda x: 1 - x[0] ** 2 - x[1] ** 2, lambda x: x[2]]


q = lambda x: np.array([x[2] - np.sqrt(x[0] ** 2 + x[1] ** 2)])

# gradient matrix as a function of x
G = lambda x: np.array(
    [
        [(-x[0] / (math.sqrt(x[0] ** 2 + x[1] ** 2)))],
        [(-x[1] / (math.sqrt(x[0] ** 2 + x[1] ** 2)))],
        [1],
    ]
)

# distribution function
f = lambda x: 1
eps = 1e-8  # error margin
nmax = 10  # number of iteration for newton solver

# initial values
x = np.array([0.1, 0.1, math.sqrt(0.1**2 + 0.1**2)])
G_x = G(x)
L_x = la.cholesky(G_x.T @ G_x, lower=True)


# ======================================================================
def newton_solver(v, G_x, L_x):
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
    # Dimension of the problem
    d = G_x.shape[1]
    a = np.zeros(d)
    y = v + np.dot(G_x, a)
    qval = q(y)
    qerr = la.norm(qval)

    i = 1
    found = False
    while i <= nmax and qerr > eps:
        # Solve for delta_a using forward-backward substitution
        delta_a = la.solve_triangular(
            L_x.T, la.solve_triangular(L_x, -qval, lower=True), lower=False
        )
        a += delta_a
        y = v + np.dot(G_x, a)
        qval = q(y)
        qerr = la.norm(qval)
        i += 1

    if i <= nmax:
        found = True

    return {"pt": y, "flag": found}


def manifold_sampler(x, G_x, L_x):
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

    d = len(x)
    z = np.random.normal(0, 1, d)
    # Compute xi
    xi = la.solve_triangular(
        L_x.T, la.solve_triangular(L_x, G_x.T @ z, lower=True), lower=False
    )
    # Tangent vector proposal
    v_x = s * (z - G_x @ xi)

    # Newton projection
    proj_for = newton_solver(x + v_x, G_x, L_x)
    if not proj_for["flag"]:
        return {"pt": x, "gradient": G_x, "chol": L_x}  # Projection failed

    y = proj_for["pt"]

    # Inequality check
    for h in constraints:
        if h(y) < 0:
            return {"pt": x, "gradient": G_x, "chol": L_x}  # Constraint violated

    # Reverse tangent proposal
    G_y = G(y)
    L_y = la.cholesky(G_y.T @ G_y, lower=True)
    delta_xy = x - y
    xi = la.solve_triangular(
        L_y.T, la.solve_triangular(L_y, G_y.T @ delta_xy, lower=True), lower=False
    )
    # tangent vector
    v_y = delta_xy - G_y @ xi

    # Normal vector
    w_y = delta_xy - v_y

    # Metropolis-Hastings step
    p_vx = math.exp(-np.linalg.norm(v_x) ** 2 / (2 * s**2))
    p_vy = math.exp(-np.linalg.norm(v_y) ** 2 / (2 * s**2))
    alpha = min(1, (f(y) * p_vy) / (f(x) * p_vx))
    u = np.random.uniform(0, 1)
    if u > alpha:
        return {"pt": x, "gradient": G_x, "chol": L_x}  # Rejection

    # Reverse projection
    proj_rev = newton_solver(y + v_y, G_y, L_y)
    if not proj_rev["flag"]:
        return {"pt": x, "gradient": G_x, "chol": L_x}  # Reverse projection failed

    # Reverse projection to a wrong point
    xx = proj_rev["pt"]
    if la.norm(xx - x) > eps:
        return {
            "pt": x,
            "gradient": G_x,
            "chol": L_x,
        }  # Reject due to distance criterion

    # return y  # Successfully sampled
    return {"pt": y, "gradient": G_y, "chol": L_y}


# ========================================================================
## test the cone case
samples = [x]
for i in range(n_iter):
    res = manifold_sampler(x, G_x, L_x)
    x = res["pt"]
    G_x = res["gradient"]
    L_x = res["chol"]
    samples.append(x)


samples = np.array(samples)

# Separate the coordinates into x, y, z
X = samples[:, 0]
Y = samples[:, 1]
Z = samples[:, 2]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot
ax.scatter(X, Y, Z, c="b", marker="o")

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Samples Scatter Plot")

# Show the plot
plt.show()
