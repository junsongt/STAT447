import numpy as np
from scipy import stats
import scipy.linalg as la
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def manifold_sampler(x, q, f, G, G_x, L_x, s, nmax, eps):
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
    v_x = s * (z - G_x @ xi)

    # guard inf case
    if np.any(np.isnan(v_x)) or np.any(np.isinf(v_x)):
        print("Invalid tangent vector v_x:", v_x)
        return {
            "pt": x,
            "grad": G_x,
            "chol": L_x,
        }  # Reject the step and return the current point

    proj_for = newton_solve(x + v_x, q, G_x, L_x, nmax, eps)
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
    v_y = delta_xy - G_y @ xi
    p_vx = math.exp(-np.linalg.norm(v_x) ** 2 / (2 * s**2))
    p_vy = math.exp(-np.linalg.norm(v_y) ** 2 / (2 * s**2))
    alpha = min(1, (f(y) * p_vy) / (f(x) * p_vx))
    u = np.random.uniform(0, 1)
    if u > alpha:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Rejection

    proj_rev = newton_solve(y + v_y, q, G_y, L_y, nmax, eps)
    if not proj_rev["flag"]:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Reverse projection failed

    xx = proj_rev["pt"]
    if la.norm(xx - x) > eps:
        return {"pt": x, "grad": G_x, "chol": L_x}  # Reject due to distance criterion

    G_x = G_y
    L_x = L_y

    return {"pt": y, "grad": G_y, "chol": L_y}


# Parameters
n_iter = 1000
sigma = np.array([1.0, 2.0])  # Diagonal covariance matrix
s = 0.05
k = 10
nmax = 10
eps = 1e-8

g = lambda x: multivariate_normal.pdf(x, mean=np.zeros_like(x), cov=np.diag(sigma))

G = lambda x: np.array(g(x) * (-x / sigma**2)).reshape(-1, 1)

# def g(x):
#     return multivariate_normal.pdf(x, mean=np.zeros_like(x), cov=np.diag(sigma))

# def G(x):
#     grad = np.array(g(x, sigma) * (-x / sigma**2))  # Gradient formula for diagonal covariance
#     return grad.reshape(-1,1)

# samples = []
# # Main algorithm
# for i in range(n_iter):
#     # Step 1: Sample an initial point x0 ~ g
#     x0 = np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag(sigma))

#     # Step 2: Compute Y = g(x0)
#     Y = g(x0)

#     # Step 3: Define the constraint q(x) = g(x) - Y
#     q = lambda x: np.array([g(x) - Y])

#     # Step 4: Define the density on the constraint
#     f = lambda x: g(x) / la.norm(G(x))

#     # Step 5: Start manifold sampling with x0
#     x = x0  # Initial point
#     G_x = G(x)
#     L_x = la.cholesky(G_x.T @ G_x, lower=True)

#     for j in range(k):
#         # Sample a point on the constraint manifold
#         res = manifold_sampler(x, q, f, G, G_x, L_x, s, nmax, eps)
#         x = res["pt"]
#         G_x = res["grad"]
#         L_x = res["chol"]

#     # Store the last sample as the i-th sample
#     samples.append(x)


def contour_sampler(manifold_ker, sigma, k, n_iter):
    samples = []
    # Main algorithm
    for i in range(n_iter):
        # Step 1: Sample an initial point x0 ~ g
        x0 = np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag(sigma))

        # Step 2: Compute Y = g(x0)
        Y = g(x0)

        # Step 3: Define the constraint q(x) = g(x) - Y
        q = lambda x: np.array([g(x) - Y])

        # Step 4: Define the density on the constraint
        f = lambda x: g(x) / la.norm(G(x))

        # Step 5: Start manifold sampling with x0
        x = x0  # Initial point
        G_x = G(x)
        L_x = la.cholesky(G_x.T @ G_x, lower=True)

        for j in range(k):
            # Sample a point on the constraint manifold
            res = manifold_sampler(x, q, f, G, G_x, L_x, s, nmax, eps)
            x = res["pt"]
            G_x = res["grad"]
            L_x = res["chol"]

        # Store the last sample as the i-th sample
        samples.append(x)
    return samples


# # test case
# samples = contour_sampler(manifold_sampler, sigma, k, n_iter)
# # Convert samples to a NumPy array for further analysis
# samples = np.array(samples)
# ref_samples = np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag(sigma), size=n_iter)


# # Create grid for contour plot of the multivariate normal density
# x_grid = np.linspace(-4, 4, 100)
# y_grid = np.linspace(-4, 4, 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# pos = np.dstack((X, Y))
# Z = multivariate_normal([0, 0], np.diag(sigma)).pdf(pos)

# # Plotting the samples
# plt.figure(figsize=(8, 6))
# plt.scatter(samples[:, 0], samples[:, 1], c="blue", alpha=0.5, label="Samples")
# plt.scatter(ref_samples[:, 0], samples[:, 1], c="green", alpha=0.5, label="Samples")
# # Plot contours of the reference multivariate normal
# contour = plt.contour(X, Y, Z, levels=10, cmap="viridis", alpha=0.7)
# # Add labels, legend, and grid
# plt.clabel(contour, inline=True, fontsize=8)
# plt.title("Samples from the manifold sampler kernel & mvnormal")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
# plt.axvline(0, color="gray", linestyle="--", linewidth=0.7)
# plt.legend(["manifold sampler kernel", "mvnormal"], loc="lower right")
# plt.show()

# print(stats.kstest(samples[:,0], ref_samples[:,0]))
# print(stats.kstest(samples[:,1], ref_samples[:,1]))


## check the joint p-values are uniform([0,1]^2)

x_pvals = []
y_pvals = []
for i in range(100):
    contour_samples = contour_sampler(manifold_sampler, sigma, k, n_iter)
    contour_samples = np.array(contour_samples)
    real_samples = np.random.multivariate_normal(
        mean=np.zeros(2), cov=np.diag(sigma), size=n_iter
    )
    p_val_x = stats.kstest(contour_samples[:, 0], real_samples[:, 0])[1]
    p_val_y = stats.kstest(contour_samples[:, 1], real_samples[:, 1])[1]
    x_pvals.append(p_val_x)
    y_pvals.append(p_val_y)


# Compute 2D histogram for samples
bins = 20  # Number of bins for the histogram
hist, x_edges, y_edges = np.histogram2d(
    x_pvals, y_pvals, bins=bins, range=[[0, 1], [0, 1]], density=True
)

# Create grid for plotting
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)

# Uniform density value for comparison
# uniform_density = np.ones_like(hist) / (bins * bins)  # Adjusted for equal-area bins
uniform_density = np.ones_like(hist)
uniform_density_scaled = uniform_density * hist.max()

# 3D Plot: Observed Histogram vs Uniform Density
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot observed density as bars
for i in range(hist.shape[0]):
    for j in range(hist.shape[1]):
        ax.bar3d(
            x_centers[i],  # x-coordinate
            y_centers[j],  # y-coordinate
            0,  # z-base
            x_edges[1] - x_edges[0],  # bar width (x)
            y_edges[1] - y_edges[0],  # bar width (y)
            hist[i, j],  # bar height
            color="blue",
            alpha=0.7,
        )

# Plot uniform density as a surface
uniform_surface = ax.plot_surface(X, Y, uniform_density_scaled, color="red", alpha=0.5)

# Customize the plot
ax.set_title("3D Comparison: Observed vs Uniform Density")
ax.set_xlabel("X-axis (x_pvals)")
ax.set_ylabel("Y-axis (y_pvals)")
ax.set_zlabel("Density/Frequency")

# Add a legend using ProxyArtists
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Define proxy artists
blue_bar = Patch(color="blue", alpha=0.7, label="Observed Histogram")
red_surface = Patch(color="red", alpha=0.5, label="Uniform Density")
ax.legend(handles=[blue_bar, red_surface], loc="upper right")

plt.show()
