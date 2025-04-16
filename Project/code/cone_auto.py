import numpy as np
from scipy import stats
import scipy.linalg as la
import math
import jax
import jax.numpy as jnp
from jax import random
import jax.scipy.linalg as jla
from manifold_sampler import Manifold_sampler
from manifold_sampler_new import Manifold_sampler_new
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================================
# cone settings
s = 0.05
n_iter = 1000  # number of samples
constraints = [lambda x: 1 - x[0] ** 2 - x[1] ** 2, lambda x: x[2]]

q = lambda x: np.array([x[2] - jnp.sqrt(x[0] ** 2 + x[1] ** 2)])

G = lambda x: np.array(
    [
        [-x[0] / jnp.sqrt(x[0] ** 2 + x[1] ** 2)],
        [-x[1] / jnp.sqrt(x[0] ** 2 + x[1] ** 2)],
        [1],
    ]
)

# distribution function
f = lambda x: 1
eps = 1e-8  # error margin
nmax = 10  # max iterations for Newton solver

# initial values
x = np.array([0.1, 0.1, math.sqrt(0.1**2 + 0.1**2)])

# sampling
sampler = Manifold_sampler_new(x, q, G, H, constraints, f, eps, nmax)
samples = [x]
for i in range(n_iter):
    x = sampler.sample(x)
    samples.append(x)

samples = np.array(samples)

# plot
X = samples[:, 0]
Y = samples[:, 1]
Z = samples[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X, Y, Z, c="b", marker="o")

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Samples Scatter Plot")
plt.show()
