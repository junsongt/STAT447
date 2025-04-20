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


# =================================================================================
def logratio(x, v_x, q, G, G_x, L_x, nmax, eps, f, s):
    proj = newton_solve(x + s * v_x, q, G_x, L_x, nmax, eps)
    while proj["flag"] == False:
        s = s / 2
        proj = newton_solve(x + s * v_x, q, G_x, L_x, nmax, eps)
    y = proj["pt"]
    G_y = G(y)
    L_y = la.cholesky(G_y.T @ G_y, lower=True)
    delta_xy = x - y
    xi = la.solve_triangular(
        L_y.T, la.solve_triangular(L_y, G_y.T @ delta_xy, lower=True), lower=False
    )
    v_y = (delta_xy - G_y @ xi) / s
    l = (
        math.log(f(y))
        - math.log(f(x))
        - (np.linalg.norm(v_y) ** 2 - np.linalg.norm(v_x) ** 2) / 2
    )
    # return {"l":l, "pt":y, "tan":v_y, "grad":G_y, "chol":L_y}
    return l


# ==================================================================================
def adjust_step(x, v_x, s, q, f, G, G_x, L_x, nmax, eps, a, b):
    l = logratio(x, v_x, q, G, G_x, L_x, nmax, eps, f, s)
    delta = (1 if abs(l) < math.log(b) else 0) - (1 if abs(l) > abs(math.log(a)) else 0)
    j = 0
    if delta == 0:
        return s
    while true:
        j = j + delta
        s = s * 2 ** (delta)
        l = logratio(x, v_x, q, G, G_x, L_x, nmax, eps, f, s)
        # delta = (1 if abs(l) < math.log(b) else 0) - (
        #     1 if abs(l) > abs(math.log(a)) else 0
        # )
        if delta == 1 and abs(l) >= abs(math.log(b)):
            return s / 2
        elif delta == -1 and abs(l) <= abs(math.log(a)):
            return s
        # else:
        #     return s


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
        # Reject the step and return the current point
        return {"pt": x, "grad": G_x, "chol": L_x}

    #  auto-adjust step size: s
    D = np.random.uniform(0, 1, 2)
    a = min(D)
    b = max(D)
    s = adjust_step(x, v_x, s, q, f, G, G_x, L_x, nmax, eps, a, b)
    print("step size:", s)

    proj_for = newton_solve(x + s * v_x, q, G_x, L_x, nmax, eps)
    if not proj_for["flag"]:
        return {"pt": x, "grad": G_x, "chol": L_x, "step": s}  # Projection failed

    y = proj_for["pt"]

    G_y = G(y)
    L_y = la.cholesky(G_y.T @ G_y, lower=True)
    delta_xy = x - y
    xi = la.solve_triangular(
        L_y.T, la.solve_triangular(L_y, G_y.T @ delta_xy, lower=True), lower=False
    )
    v_y = (delta_xy - G_y @ xi) / s
    # p_vx = math.exp(-np.linalg.norm(v_x)**2 / 2)
    # p_vy = math.exp(-np.linalg.norm(v_y)**2 / 2)
    # alpha = min(1, (f(y) * p_vy) / (f(x) * p_vx))
    alpha = min(
        1,
        math.exp(
            math.log(f(y))
            - math.log(f(x))
            - (np.linalg.norm(v_y) ** 2 - np.linalg.norm(v_x) ** 2) / 2
        ),
    )
    # print("alpha: ", alpha)
    u = np.random.uniform(0, 1)
    # M-H Rejection
    if u > alpha:
        print("M-H rejection!")
        return {"pt": x, "grad": G_x, "chol": L_x, "step": s}

    proj_rev = newton_solve(y + s * v_y, q, G_y, L_y, nmax, eps)
    if not reversible(proj_rev, x, eps):
        print("reversibility rejection!")
        return {"pt": x, "grad": G_x, "chol": L_x, "step": s}
    # # Reverse projection failed
    # if not proj_rev["flag"]:
    #     return {"pt": x,"grad": G_x,"chol": L_x,"step": s}

    # xx = proj_rev["pt"]
    # # Reject due to distance criterion
    # if la.norm(xx - x) > eps:
    #     return {"pt": x,"grad": G_x,"chol": L_x,"step": s}
    print("proposal success!")
    G_x = G_y
    L_x = L_y

    return {"pt": y, "grad": G_y, "chol": L_y, "step": s}


def reversible(solve_obj, x, eps):
    # Reverse projection failed
    if solve_obj["flag"] == False:
        # print("reverse projection failed!")
        return False
    xx = solve_obj["pt"]
    # Reject due to convergence at wrong point
    if la.norm(xx - x) > eps:
        # print("convergence at wrong point!")
        return False
    # print("reversibility check passed!")
    return True


# # ==============================================================================
# # ellipsoid settings
# s = 0.5
# A = 2
# B = 1
# C = 1.5

# n_iter = 1000  # number of samples
# # constraints are {h_i(x) : h_i(x) >= 0}
# q = lambda x: np.array([x[0] ** 2 / A**2 + x[1] ** 2 / B**2 + x[2] ** 2 / C**2 - 1])
# # gradient matrix as a function of x
# G = lambda x: np.array([[2 * x[0] / A**2], [2 * x[1] / A**2], [2 * x[2] / A**2]])

# # distribution function
# f = lambda x: 1
# eps = 1e-4  # error margin
# nmax = 10  # number of iteration for newton solver
# # initial values
# x = np.array([2, 0, 0])
# G_x = G(x)
# L_x = la.cholesky(G_x.T @ G_x, lower=True)

# # # unit test
# # d = len(x)
# # z = np.random.normal(0, 1, d)
# # xi = la.solve_triangular(L_x.T, la.solve_triangular(L_x, G_x.T @ z, lower=True),lower=False)
# # v_x = z - G_x @ xi
# # D = np.random.uniform(0, 1, 2)
# # a = min(D)
# # b = max(D)
# # print(adjust_step(x, v_x, s, q, f, G, G_x, L_x, nmax, eps, a, b))

# # =======================================================================
# # sampling
# samples = [x]
# prev = x
# count = 0
# for i in range(n_iter):
#     res = sample(x, q, f, G, G_x, L_x, s, nmax, eps)
#     x = res["pt"]
#     if not np.array_equal(x, prev):
#         count = count + 1
#     samples.append(x)
#     G_x = res["grad"]
#     L_x = res["chol"]
#     s = res["step"]

# samples = np.array(samples)
# print(
#     "number of samples: ", len(samples), " and the proportion: ", count / len(samples)
# )

# # plot
# # Separate the coordinates into x, y, z
# X = samples[:, 0]
# Y = samples[:, 1]
# Z = samples[:, 2]

# # Create the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# ###########################################
# # final plot mode
# # Scatter plot
# ax.scatter(X, Y, Z, c="b", marker="o")

# # Labels and title
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
# ax.set_title("ellipsoid")

# ###########################################
# # # animation mode
# # scatter = ax.scatter([], [], [], c="b", marker="o")

# # # Set axes labels
# # ax.set_xlabel("X-axis")
# # ax.set_ylabel("Y-axis")
# # ax.set_zlabel("Z-axis")
# # ax.set_title("ellipsoid animation")

# # # Set axis limits
# # ax.set_xlim([min(X) - 0.1, max(X) + 0.1])
# # ax.set_ylim([min(Y) - 0.1, max(Y) + 0.1])
# # ax.set_zlim([min(Z) - 0.1, max(Z) + 0.1])


# # # Update function for animation
# # def update(frame):
# #     scatter._offsets3d = (X[:frame], Y[:frame], Z[:frame])
# #     return (scatter,)


# # # Create the animation object
# # ani = animation.FuncAnimation(
# #     fig, update, frames=n_iter, interval=50, blit=False, repeat=False
# # )
# # ani.save("D:/ellipsoid_animation.mp4", writer="ffmpeg")

# ###########################################

# # Show the plot
# plt.show()
