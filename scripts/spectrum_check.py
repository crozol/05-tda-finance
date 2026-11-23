"""What the topology of a standardised window can see: a numerical check.

Proposition. Let the n days of a window be independent draws from N(mu, Sigma) in R^d,
n > d, and let Z be the window standardised per asset with its own sample mean and
standard deviation (src.persistence.standardize_windows). Then

    Z Z^T = n V diag(lam) V^T,

where lam are the eigenvalues of the sample correlation matrix of the window and V,
an n x d matrix with orthonormal columns orthogonal to the vector of ones, is uniformly
distributed on that set and independent of lam.

Proof. Write the centred window as A = P X, P = I - 11^T/n. Its law is invariant under
the rotations of R^n that fix the vector of ones, which leave A^T A unchanged, so its
polar factor A (A^T A)^(-1/2) is uniform given A^T A. The scales of the standardisation
are a function of A^T A, and Z Z^T = A D^-1 A^T has the eigenvalues of D^-1/2 A^T A D^-1/2
= n R_hat, with eigenvectors fixed by A^T A; multiplying the uniform frame by them leaves
it uniform.

Consequence. Every Vietoris-Rips diagram of the days is a function of the distances
between them, that is of Z Z^T. Given the sample correlation spectrum, its law does not
depend on mu or Sigma: if the days of each window are Gaussian, a regime that changes the
mean, the volatility or the correlation reaches the topology only through the spectrum,
and topology adds nothing to it. It can add something only through what the proposition
assumes away: heavy tails, or days of one window that do not share a law (a regime
change inside the window).

The check. For each window, a spectral surrogate replaces the frame V by a fresh uniform
one and keeps the window's own spectrum. Under the proposition the window and its
surrogate are exchangeable given lam, so the paired difference of any statistic of
their diagrams is symmetric about zero (Wilcoxon signed-rank test). Five synthetic
scenarios, 40-day windows of 7 assets as in the analysis plan: three Gaussian ones
(calm; crisis with a negative mean; a random correlation matrix with unequal scales),
where the test should find nothing, and two that break the assumptions (Student t with
3 degrees of freedom, and a regime change halfway through the window), where it should.

    python -m scripts.spectrum_check          # -> results/spectrum_check.json, figures/spectrum_check.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_DAYS, N_ASSETS = 40, 7
STATS = ("L1, H0", "L1, H1", "L2, H0", "L2, H1")   # column order of src.features.landscape_norms


def one_factor(rho: float, d: int = N_ASSETS) -> np.ndarray:
    return rho * np.ones((d, d)) + (1 - rho) * np.eye(d)


def gaussian_windows(rng, n_windows: int, corr: np.ndarray, scales=None, mean=None) -> np.ndarray:
    d = corr.shape[0]
    x = rng.normal(size=(n_windows, N_DAYS, d)) @ np.linalg.cholesky(corr).T
    if scales is not None:
        x = x * scales
    if mean is not None:
        x = x + mean
    return x


def scenarios(rng, n_windows: int) -> dict[str, np.ndarray]:
    from scipy.stats import random_correlation

    calm, crisis = one_factor(0.3), one_factor(0.7)
    spectrum = np.array([4.2, 0.9, 0.6, 0.45, 0.35, 0.3, 0.2])          # sums to 7
    random_corr = random_correlation.rvs(spectrum, random_state=rng)
    scales = np.linspace(0.005, 0.03, N_ASSETS)
    w = rng.chisquare(3, size=(n_windows, N_DAYS, 1))
    half = N_DAYS // 2
    switch = np.concatenate(
        [gaussian_windows(rng, n_windows, calm)[:, :half] * 0.01,
         gaussian_windows(rng, n_windows, crisis)[:, half:] * 0.02], axis=1)
    return {
        "Gaussian, calm": 0.01 * gaussian_windows(rng, n_windows, calm),
        "Gaussian, crisis with a negative mean": gaussian_windows(rng, n_windows, crisis, 0.02, -0.004),
        "Gaussian, random correlation, unequal scales": gaussian_windows(rng, n_windows, random_corr, scales),
        "Student t, 3 degrees of freedom": 0.01 * gaussian_windows(rng, n_windows, calm) * np.sqrt(1 / w),
        "regime change halfway through the window": switch,
    }


def uniform_frames(rng, n_windows: int, n: int = N_DAYS, d: int = N_ASSETS) -> np.ndarray:
    """Uniformly distributed n x d frames with orthonormal columns orthogonal to the vector
    of ones: the polar factor of a centred Gaussian matrix."""
    g = rng.normal(size=(n_windows, n, d))
    g -= g.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    return u @ vt


def spectral_surrogates(z: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """For standardised windows z (n_windows, n, d): the sample correlation spectra, and
    windows sqrt(n) V diag(sqrt(lam)) with a fresh uniform frame V, whose Gram matrix is
    n V diag(lam) V^T, the law the proposition gives the window's own."""
    n = z.shape[1]
    lam = np.clip(np.linalg.eigvalsh(np.einsum("wti,wtj->wij", z, z) / n), 0.0, None)
    return lam, np.sqrt(n) * uniform_frames(rng, len(z), n, z.shape[2]) * np.sqrt(lam)[:, None, :]


def run(n_windows: int = 1000, seed: int = 0) -> dict:
    from scipy.stats import wilcoxon

    from src.features import landscape_norms
    from src.persistence import persistence_diagrams, standardize_windows

    rng = np.random.default_rng(seed)
    real = {k: standardize_windows(v) for k, v in scenarios(rng, n_windows).items()}
    surrogate, spectra = {}, {}
    for k, z in real.items():
        spectra[k], surrogate[k] = spectral_surrogates(z, rng)
    # one call, so that every diagram is padded alike and every norm uses the same grid
    stacked = np.concatenate([*real.values(), *surrogate.values()])
    norms = landscape_norms(persistence_diagrams(stacked, (0, 1)))
    names = list(real)
    out = {"n_windows": n_windows, "n_days": N_DAYS, "n_assets": N_ASSETS, "seed": seed,
           "statistics": list(STATS), "scenarios": {}}
    for i, name in enumerate(names):
        a = norms[i * n_windows:(i + 1) * n_windows]
        b = norms[(len(names) + i) * n_windows:(len(names) + i + 1) * n_windows]
        diff = a - b
        per = {}
        for j, s in enumerate(STATS):
            d = diff[:, j]
            per[s] = {"real_mean": float(a[:, j].mean()), "surrogate_mean": float(b[:, j].mean()),
                      "mean_difference": float(d.mean()),
                      "se_difference": float(d.std(ddof=1) / np.sqrt(len(d))),
                      "standardised_difference": float(d.mean() / a[:, j].std(ddof=1)),
                      "wilcoxon_p": float(wilcoxon(d).pvalue)}
        out["scenarios"][name] = {
            "gaussian": name.startswith("Gaussian"),
            "leading_eigenvalue_share_mean": float(spectra[name][:, -1].mean() / N_ASSETS),
            "statistics": per,
            "paired_differences": {s: diff[:, j].round(6).tolist() for j, s in enumerate(STATS)}}
    return out


def plot(res: dict, out: str = "figures/spectrum_check.png") -> None:
    import matplotlib.pyplot as plt

    from scripts.make_figures import AMBER, CYAN, FG_1, PINK, style

    style()
    names = list(res["scenarios"])
    labels = {"Gaussian, calm": "Gaussian\ncalm", "Gaussian, crisis with a negative mean": "Gaussian\ncrisis,\nmean < 0",
              "Gaussian, random correlation, unequal scales": "Gaussian\nrandom R,\nscales",
              "Student t, 3 degrees of freedom": "Student t\n3 d.o.f.",
              "regime change halfway through the window": "regime\nchange in\nwindow"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, stat in zip(axes, ("L1, H0", "L1, H1")):
        data, colors = [], []
        for name in names:
            sc = res["scenarios"][name]
            d = np.asarray(sc["paired_differences"][stat])
            data.append(d / np.std(d, ddof=1) if sc["gaussian"] else d / np.std(d, ddof=1))
            colors.append(CYAN if sc["gaussian"] else PINK)
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.35)
        for k, (name, d) in enumerate(zip(names, data), start=1):
            st = res["scenarios"][name]["statistics"][stat]
            m, se = st["mean_difference"], st["se_difference"]
            z = m / se
            ax.plot([k], [np.mean(d)], "o", color=colors[k - 1], ms=6, zorder=3)
            p = st["wilcoxon_p"]
            ax.annotate(f"p = {p:.2f}" if p >= 0.01 else (f"p = {p:.0e}" if p > 0 else "p < 1e-300"),
                        (k, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=FG_1 if p >= 0.05 else AMBER)
        ax.axhline(0, color=FG_1, lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(range(1, len(names) + 1), [labels[n] for n in names], fontsize=8.5)
        dim = "H₀ (components)" if stat.endswith("H0") else "H₁ (loops)"
        ax.set_title(f"landscape L¹ norm, {dim}", fontsize=11, loc="left", pad=10)
        ax.set_ylim(-4.5, 5.5)
    axes[0].set_ylabel("window − its spectral surrogate (in std)", fontsize=9.5)
    fig.suptitle("Given its correlation spectrum, a Gaussian window's topology is pure noise",
                 x=0.06, ha="left", y=1.02, fontsize=13, fontweight="bold")
    fig.text(0.06, -0.06, "paired difference of a statistic between each window and a surrogate with the same "
             "sample correlation spectrum; Wilcoxon signed-rank p-values on 1 000 windows", fontsize=9, color=FG_1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/spectrum_check.json")
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved results")
    a = p.parse_args()
    if a.plot_only:
        plot(json.loads(Path(a.out).read_text()))
        raise SystemExit
    res = run(a.windows, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, sc in res["scenarios"].items():
        print(f"{name:48s} " + "  ".join(
            f"{s}: d={v['standardised_difference']:+.3f} p={v['wilcoxon_p']:.2g}" for s, v in sc["statistics"].items()))
    plot(res)
    print(f"[ok] {a.out}")
