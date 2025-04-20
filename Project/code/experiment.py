"""
benchmark_ellipse_manifold.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compare `manifold.py` and `manifold_auto.py`
on an extremely thin 2D ellipse.

Produces
--------
1. 3D trace plot for each sampler
2. Autocorrelation plot (x-coordinate, first 50 lags)
3. ESS versus number of samples
4. Wall-clock time versus number of samples
5. Pandas table with ESS & time statistics
"""

import time, io, contextlib, importlib.util, sys, types
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------
# OPTIONAL – patch JAX imports away if JAX isn't installed
jax_stub = types.ModuleType("jax")
jax_stub.random = types.ModuleType("random")
jax_stub.numpy = types.ModuleType("numpy")
jax_stub.grad = lambda *a, **k: None
jax_stub.value_and_grad = lambda *a, **k: None
sys.modules.setdefault("jax", jax_stub)
sys.modules.setdefault("jax.random", jax_stub.random)
sys.modules.setdefault("jax.numpy", jax_stub.numpy)


# ------------------------------------------------------------------
def load_module(name: str, path: str, dummy: str | None = None):
    """Silently import a sampler file that may print or run a demo."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if dummy:  # ensure placeholder for top‑level demo call
        mod.__dict__[dummy] = lambda *a, **k: None
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # ignore failures triggered by demo‐code at the end of the file
            pass
    return mod


manifold = load_module("manifold_sampler", "manifold.py")
manifold_a = load_module(
    "manifold_auto_sampler", "manifold_auto.py", dummy="manifold_auto"
)

# ------------------------------------------------------------------
# Problem definition – thin ellipse  x²/a² + y²/b² = 1  with b ≪ a
A, B = 1.0, 0.1  # semi axes
q = lambda x: np.array([x[0] ** 2 / A**2 + x[1] ** 2 / B**2 - 1])
G = lambda x: np.array([[2 * x[0] / A**2], [2 * x[1] / B**2]])  # Jacobian of q
f = lambda x: 1.0  # uniform on the manifold
eps, nmax = 1e-6, 10

x0 = np.array([A, 0.0])  # valid starting point
Gx0 = G(x0)
Lx0 = np.linalg.cholesky(Gx0.T @ Gx0)
s0 = 0.1  # initial step size


# ------------------------------------------------------------------
def run_chain(sample_fn, n, s_init=0.1):
    """Run `sample_fn` for *n* iterations – mute its prints."""
    x, Gx, Lx, s = x0.copy(), Gx0.copy(), Lx0.copy(), s_init
    samples = []
    with contextlib.redirect_stdout(io.StringIO()):
        tic = time.perf_counter()
        for _ in range(n):
            res = sample_fn(x, q, f, G, Gx, Lx, s, nmax, eps)
            x, Gx, Lx = res["pt"], res["grad"], res["chol"]
            s = res.get("step", s)
            samples.append(x.copy())
        elapsed = time.perf_counter() - tic
    return np.asarray(samples), elapsed


# ------------------------------------------------------------------
def autocorr(series, max_lag=50):
    """Biased ACF (sufficient for visual comparison)."""
    series = np.asarray(series) - np.mean(series)
    var = np.var(series)
    acf = [1.0]
    for k in range(1, max_lag + 1):
        acf.append(float(np.dot(series[:-k], series[k:]) / ((len(series) - k) * var)))
    return np.asarray(acf)


def ess(series):
    """ESS using the initial‑positive sequence estimator."""
    ac = autocorr(series, len(series) - 2)
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        tau += 2 * ac[k]
    return len(series) / tau


# ------------------------------------------------------------------
N_TOTAL = 1000
chain_v, _ = run_chain(manifold.sample, N_TOTAL, s0)
chain_a, _ = run_chain(manifold_a.sample, N_TOTAL, s0)

# ------------------------------------------------------------------ PLOTS
plt.style.use("seaborn-v0_8-whitegrid")


# 1. Trace plots -----------------------------------------------------
def trace_plot(chain, title, save_as):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(chain[:, 0], chain[:, 1], np.arange(len(chain)), lw=0.8)
    ax.set(xlabel="x", ylabel="y", zlabel="iteration", title=title)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150)
    plt.close(fig)


trace_plot(chain_v, "Manifold sampler – trace", "trace_manifold.png")
trace_plot(chain_a, "Auto‑step sampler – trace", "trace_auto.png")

# 2. Autocorrelation -------------------------------------------------
MAX_LAG = 50
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(MAX_LAG + 1), autocorr(chain_v[:, 0], MAX_LAG), label="manifold")
ax.plot(range(MAX_LAG + 1), autocorr(chain_a[:, 0], MAX_LAG), label="auto")
ax.set(xlabel="lag", ylabel="ACF", title="Autocorrelation (x‑coord)")
ax.legend()
fig.tight_layout()
fig.savefig("acf.png", dpi=150)
plt.close(fig)

# 3. ESS + 4. Runtime versus N --------------------------------------
grid = [10, 50, 100, 500, 1_000]
ess_v = []
ess_a = []
t_v = []
t_a = []
for n in grid:
    ess_v.append(ess(chain_v[:n, 0]))
    ess_a.append(ess(chain_a[:n, 0]))
    _, tv = run_chain(manifold.sample, n)
    _, ta = run_chain(manifold_a.sample, n)
    t_v.append(tv)
    t_a.append(ta)

# ESS plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(grid, ess_v, marker="o", label="manifold")
ax.plot(grid, ess_a, marker="o", label="auto")
ax.set(xlabel="samples", ylabel="ESS (x‑coord)", title="ESS vs N")
ax.legend()
fig.tight_layout()
fig.savefig("ess_vs_n.png", dpi=150)
plt.close(fig)

# Runtime plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(grid, t_v, marker="o", label="manifold")
ax.plot(grid, t_a, marker="o", label="auto")
ax.set(xlabel="samples", ylabel="time (s)", title="Runtime vs N")
ax.legend()
fig.tight_layout()
fig.savefig("time_vs_n.png", dpi=150)
plt.close(fig)

# 5. Summary table ---------------------------------------------------
summary = pd.DataFrame(
    {
        "N": grid,
        "ESS_manifold": ess_v,
        "ESS_auto": ess_a,
        "time_manifold(s)": t_v,
        "time_auto(s)": t_a,
    }
)
print(summary.to_string(index=False))

# ------------------------------------------------------------------
print("\nPNG files written:")
for fname in [
    "trace_manifold.png",
    "trace_auto.png",
    "acf.png",
    "ess_vs_n.png",
    "time_vs_n.png",
]:
    print("  •", fname)
