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
import jax.scipy.linalg as jla
import math
from manifold_sampler import Manifold_sampler

# ================================================================
# cone settings
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

# sampling
sampler = Manifold_sampler(x, q, G, constraints, s, f, eps, nmax)
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
# ax.set_title("cone")


###########################################
# animation mode
scatter = ax.scatter([], [], [], c="b", marker="o")

# Set axes labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("cone animation")

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
ani.save("D:/cone_animation.mp4", writer="ffmpeg")

###########################################
# Show the plot
plt.show()
