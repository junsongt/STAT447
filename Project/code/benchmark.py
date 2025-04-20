# benchmark.py
# -----------------------------------------------------------
# Toggle the sections to run
RUN_TRACE = True  # dual‑inital 3‑D trace plot (mixing)
RUN_ACF = False  # ACF – both line & bar
RUN_ESS = False  # ESS versus N
RUN_RUNTIME = False  # wall‑clock time versus N
RUN_SAMPLING_EFFECT = False
# -----------------------------------------------------------

import time, io, contextlib, importlib.util, sys, types
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

np.random.seed(1)


# --- helper to load samplers silently ---------------
def _load(name, path, dummy=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if dummy:
        mod.__dict__[dummy] = lambda *a, **k: None  # spare demo code
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


manifold = _load("manifold_sampler", "manifold.py")
auto = _load("manifold_auto_sampler", "manifold_auto.py", "manifold_auto")

# --- thin ellipse target  ----------------------------------
A, B = 1.0, 0.1
q = lambda x: np.array([x[0] ** 2 / A**2 + x[1] ** 2 / B**2 - 1])
G = lambda x: np.array([[2 * x[0] / A**2], [2 * x[1] / B**2]])
f = lambda x: 1.0  # uniform on the manifold
eps, nmax = 1e-6, 10
s0 = 0.2  # initial step

x_east = np.array([A, 0.0])
x_west = np.array([-A, 0.0])
Gx0 = G(x_east)

Lx0 = la.cholesky(Gx0.T @ Gx0, lower=True)


#-----------------helpers---------------------------
def _run_chain(sample_fn, n, x0, s_init):
    # --- START‑POINT‑SPECIFIC geometry --------------------
    Gx  = G(x0)                              # fresh Jacobian
    Lx  = la.cholesky(Gx.T @ Gx, lower=True) # Cholesky of G^T G
    x   = x0.copy()
    s   = s_init
    pts = []
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            res = sample_fn(x, q, f, G, Gx, Lx, s, nmax, eps)
            x, Gx, Lx = res["pt"], res["grad"], res["chol"]
            s = res.get("step", s)
            pts.append(x.copy())
    return np.asarray(pts), time.perf_counter() - t0


def _autocorr(series, max_lag=50):
    s = series - series.mean()
    v = np.var(s)
    ac = [1.0]
    for k in range(1, max_lag + 1):
        ac.append(float(np.dot(s[:-k], s[k:]) / ((len(s) - k) * v)))
    return np.asarray(ac)


def _ess(series):
    ac = _autocorr(series, len(series) - 2)
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        tau += 2 * ac[k]
    return len(series) / tau


# ===========================================================
# 1. 3‑D dual‑initial traceplot
# ===========================================================
if RUN_TRACE:
    N = 1000
    for sampler, name in [(manifold.sample, "manifold"), (auto.sample, "manifold_auto")]:
        c1, _ = _run_chain(sampler, N, x_east, s0)
        c2, _ = _run_chain(sampler, N, x_west, s0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        it = np.arange(N)

        lbl1 = fr"start (+1, 0)  $s_0={s0}$"
        lbl2 = fr"start (-1, 0)  $s_0={s0}$"
        ax.plot(c1[:, 0], c1[:, 1], it, lw=0.8, label=lbl1)
        ax.plot(c2[:, 0], c2[:, 1], it, lw=0.8, label=lbl2)

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="iteration",
            title=f"{name}: traceplots of two chains on a thin ellipse",
        )
        ax.legend()

        # Point the *iteration* axis toward the viewer
        ax.view_init(elev=25, azim=-75)
        # ax.view_init(elev=0, azim=90)
        plt.tight_layout()
        plt.show()  # interactive – rotate & save manually

# ===========================================================
# 2. ACF plot (line + bar)
# ===========================================================
if RUN_ACF:
    N = 1000
    cv, _ = _run_chain(manifold.sample, N, x_east, s0)
    ca, _ = _run_chain(auto.sample, N, x_east, s0)
    LAG = 50
    av, aa = _autocorr(cv[:, 0], LAG), _autocorr(ca[:, 0], LAG)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6, 6), sharex=True, constrained_layout=True
    )

    lbl1 = fr"manifold  $s_0={s0}$"
    lbl2 = fr"manifold_auto  $s_0={s0}$"
    # line plot
    ax1.plot(range(LAG + 1), av, label=lbl1)
    ax1.plot(range(LAG + 1), aa, label=lbl2)
    ax1.set(title="ACF(line)", ylabel="rho(k)")
    ax1.legend()

    # bar / stem plot
    ax2.bar(range(LAG + 1), av, width=0.8, alpha=0.6, label=lbl1)
    ax2.bar(range(LAG + 1), aa, width=0.8, alpha=0.4, label=lbl2)
    ax2.set(title="ACF(bar)", xlabel="lag k", ylabel="rho(k)")
    ax2.legend()
    plt.show()

# ===========================================================
# 3. ESS vs N
# ===========================================================
if RUN_ESS:
    grid = [10, 50, 100, 500, 1000]
    cv, _ = _run_chain(manifold.sample, max(grid), x_east, s0)
    ca, _ = _run_chain(auto.sample, max(grid), x_east, s0)

    ess_v, ess_a = [], []
    for n in grid:
        ess_v.append(_ess(cv[:n, 0]))
        ess_a.append(_ess(ca[:n, 0]))

    lbl1 = fr"manifold  $s_0={s0}$"
    lbl2 = fr"manifold_auto  $s_0={s0}$"

    plt.figure()
    plt.plot(grid, ess_v, marker="o", label=lbl1)
    plt.plot(grid, ess_a, marker="o", label=lbl2)
    plt.xlabel("number of samples")
    plt.ylabel("ESS (x-coordinate)")
    plt.title("Effective Sample Size vs N")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===========================================================
# 4. Wall‑clock time vs N
# ===========================================================
if RUN_RUNTIME:
    grid = [10, 50, 100, 500, 1000]
    times_v, times_a = [], []
    for n in grid:
        _, tv = _run_chain(manifold.sample, n, x_east, s0)
        _, ta = _run_chain(auto.sample, n, x_east, s0)
        times_v.append(tv)
        times_a.append(ta)
    lbl1 = fr"manifold  $s_0={s0}$"
    lbl2 = fr"manifold_auto  $s_0={s0}$"

    plt.figure()
    plt.plot(grid, times_v, marker="o", label=lbl1)
    plt.plot(grid, times_a, marker="o", label=lbl2)
    plt.xlabel("number of samples")
    plt.ylabel("wall‑clock time (s)")
    plt.title("Runtime vs N")
    plt.legend()
    plt.tight_layout()
    plt.show()


if RUN_SAMPLING_EFFECT:
    N_TOTAL = 2000 # draw plenty; keep ~10 s run‑time
    BURN    = 200 # discard early transient

    pts_v, _ = _run_chain(manifold.sample, N_TOTAL, x_east, s0)
    pts_a, _ = _run_chain(auto.sample, N_TOTAL, x_east, s0)
    pts_v, pts_a = pts_v[BURN:], pts_a[BURN:]

    # parametric ellipse for reference
    t = np.linspace(0, 2*np.pi, 400)
    ex, ey = A*np.cos(t), B*np.sin(t)

    fig, axes = plt.subplots(2, 1, figsize=(9,4), sharex=True, sharey=True)

    for ax, data, lab in zip(
            axes, [pts_v, pts_a], ["manifold", "autostep"]):

        ax.plot(ex, ey, "k--", lw=1) # ideal curve
        ax.scatter(data[:,0], data[:,1],
                   s=10, alpha=0.5) # sampled points

        ax.set_title(lab)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal") # keep ellipse shape

    fig.suptitle("Coverage of a very thin ellipse after burn-in", y=1.02)
    plt.tight_layout()
    plt.show()