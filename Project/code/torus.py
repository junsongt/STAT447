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
from sympy import symbols, diff, hessian
from manifold_sampler import Manifold_sampler

# ================================================================
# torus settings
s = 0.1
R = 1
r = 0.4

n_iter = 1000  # number of samples
# constraints are {h_i(x) : h_i(x) >= 0}
constraints = []
q = lambda x: np.array([(R - np.sqrt(x[0] ** 2 + x[1] ** 2)) ** 2 + x[2] ** 2 - r**2])


# GPT said: To prevent potential overflow, add checks in your `q` function:
def safe_q(x):
    # Check if x[0] and x[1] are close enough to prevent division by zero or extremely large numbers
    norm_xy = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if norm_xy < 1e-10:  # prevent division by zero or extremely small values
        return np.array([float("inf")])  # return inf to indicate invalid point

    return np.array([(R - norm_xy) ** 2 + x[2] ** 2 - r**2])


# gradient matrix as a function of x
G = lambda x: np.array(
    [
        [
            -2
            * (R - np.sqrt(x[0] ** 2 + x[1] ** 2))
            * x[0]
            / np.sqrt(x[0] ** 2 + x[1] ** 2)
        ],
        [
            -2
            * (R - np.sqrt(x[0] ** 2 + x[1] ** 2))
            * x[1]
            / np.sqrt(x[0] ** 2 + x[1] ** 2)
        ],
        [2 * x[2]],
    ]
)

# distribution function
f = lambda x: 1
eps = 1e-8  # error margin
nmax = 10  # number of iteration for newton solver

# initial values
x = np.array([R + r, 0, 0])
# G_x = G(x)
# L_x = la.cholesky(G_x.T @ G_x, lower=True)

# sampling
sampler = Manifold_sampler(x, q, G, constraints, s, f, eps, nmax=100)
samples = [x]
for i in range(n_iter):
    x = sampler.sample(x)
    samples.append(x)

samples = np.array(samples)

# plot
# Separate the coordinates into x, y, z
X = samples[:, 0]
Y = samples[:, 1]
Z = samples[:, 2]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

###########################################
# # final plot mode
# # Scatter plot
# ax.scatter(X, Y, Z, c="b", marker="o")

# # Labels and title
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
# ax.set_title("torus")


###########################################
# animation mode
scatter = ax.scatter([], [], [], c="b", marker="o")

# Set axes labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("torus animation")

# Set axis limits
ax.set_xlim([min(X) - 0.1, max(X) + 0.1])
ax.set_ylim([min(Y) - 0.1, max(Y) + 0.1])
ax.set_zlim([min(Z) - 0.1, max(Z) + 0.1])


# Update function for animation
def update(frame):
    scatter._offsets3d = (X[:frame], Y[:frame], Z[:frame])
    return (scatter,)


# Create the animation object
ani = animation.FuncAnimation(
    fig, update, frames=n_iter, interval=50, blit=False, repeat=False
)
ani.save("D:/torus_animation.mp4", writer="ffmpeg")

###########################################


# Show the plot
plt.show()
