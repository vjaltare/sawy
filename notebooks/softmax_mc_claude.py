"""
Corrected plotting for the softmax / Jacobian Monte-Carlo sweeps.

FIXES vs. the original scripts
------------------------------
F1  INSET DUPLICATION (the big one).  cos_sim_q_s does not depend on nu or on
    scale_factor, but it is appended inside both loops. Every (C, resample) is
    therefore repeated 10x in the softmax dataframe and 50x in the Jacobian
    dataframe, so seaborn's bootstrap CI is too narrow by ~3.2x and ~7.1x
    respectively. Fix: drop_duplicates on (num_classes, resample) first.

F2  MAIN-PANEL DUPLICATION (Jacobian only).  Cosine is scale-invariant, so all
    five scale_factor rows carry the same value (spread across sf has median
    3.6e-7, i.e. float32 rounding). CI too narrow by ~2.24x. Fix: filter to a
    single scale_factor, as already done for the scatter.

F3  PALETTE SOURCE.  The logistic figures built their palette from the GAUSSIAN
    softmax dataframe. Works only because both happen to have 10 class values.

F4  GUIDE LINE.  The x/3 line is now measured from the data
    (slope = mean of norm_ratio_z_q / norm_ratio_z_s) rather than hardcoded.

F5  ERRORBAR made explicit ("sd" over resamples) instead of seaborn's default
    bootstrap CI, so the caption can state what the band is.

Not fixed here (requires a re-run):
    The gaussian sweep used nu = 2..1024 (powers of 2); the logistic sweep used
    nu = 2..49999 (log-spaced). Side-by-side left panels are not comparable.
    Either re-run one grid or pass nu_max=1024 to truncate the logistic set.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'


LEVEL_CFG = {
    "softmax": dict(sym=r"\hat{s}", tgt="s", inset_tgt=r"\hat{s}_\infty"),
    "jacobian": dict(sym=r"\hat{J}", tgt="J", inset_tgt=r"\hat{J}_\infty"),
}


def _dedup_inset(df: pd.DataFrame) -> pd.DataFrame:
    """F1: one row per (num_classes, resample) for window-independent metrics."""
    return df.drop_duplicates(subset=["num_classes", "resample"])


def _one_sf(df: pd.DataFrame) -> pd.DataFrame:
    """F2: cosine is scale-invariant -> keep a single scale_factor."""
    if "scale_factor" not in df.columns:
        return df
    sf = 1.0 if np.any(np.isclose(df["scale_factor"], 1.0)) else df["scale_factor"].min()
    return df[np.isclose(df["scale_factor"], sf)]


def measured_slope(df: pd.DataFrame) -> float:
    """F4: slope of the norm scatter = ||target|| / ||estimator_infty||.
    nu and scale_factor both cancel, so this is purely the link gap."""
    d = _one_sf(df)
    return float((d["norm_ratio_z_q"] / d["norm_ratio_z_s"]).mean())


def plot_level(
    df: pd.DataFrame,
    level: str = "jacobian",
    palette: str = "flare",
    nu_max: int | None = None,
    show_slope_guide: bool = True,
    lim: float = 20.0,
    figsize=(7.5, 3.5),
    errorbar=("pi", 68),
):
    """errorbar: ("pi", 68) is a percentile interval -- respects the [0,1]
    bound of a cosine. "sd" is honest but draws bands outside [0,1]; seaborn's
    default bootstrap CI is fine too, but state whichever you use in the
    caption."""
    cfg = LEVEL_CFG[level]
    if nu_max is not None:                      # optional: match nu grids
        df = df[df["int_window"] <= nu_max]

    n_c = df["num_classes"].nunique()
    pal = sns.color_palette(palette, n_c)       # F3: palette from THIS dataframe

    cos_df = _one_sf(df).copy()                 # F2
    cos_df["num_classes"] = pd.Categorical(     # discrete legend, not continuous
        cos_df["num_classes"], sorted(cos_df["num_classes"].unique()), ordered=True)

    fig, ax = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    # ---------------- left: convergence to the infinite-window estimator -----
    sns.lineplot(data=cos_df, x="int_window", y="cos_sim_z_q", hue="num_classes",
                 lw=1.5, alpha=0.6, ax=ax[0], palette=pal,
                 errorbar=errorbar)             # F5
    ax[0].set_xscale("log", base=10)
    ax[0].set_ylabel(rf"cos-sim$({cfg['sym']}_\nu,\ {cfg['sym']}_\infty)$",
                     fontsize=18)
    ax[0].set_xlabel(r"$\nu$ (samples)", fontsize=18)
    ax[0].legend().set_visible(False)

    # inset: the link gap (window-independent -> deduplicate)   F1
    inset_df = _dedup_inset(df)
    axi = ax[0].inset_axes([0.55, 0.15, 0.35, 0.35])
    sns.lineplot(data=inset_df, x="num_classes", y="cos_sim_q_s", ax=axi,
                 alpha=0.7, color=pal[0], lw=2, marker="o", errorbar=errorbar)
    axi.set_xscale("log", base=10)
    axi.axhline(1.0, color="0.5", alpha=0.7, lw=1.5, ls="--")
    axi.set_ylabel(rf"sim$({cfg['inset_tgt']},\ {cfg['tgt']})$", fontsize=14)
    axi.set_xlabel("# Classes", fontsize=14)
    axi.set_ylim(0.5, 1.01)

    # ---------------- right: norm scatter ------------------------------------
    sc_df = _one_sf(df).copy()
    sc_df["num_classes"] = pd.Categorical(
        sc_df["num_classes"], sorted(sc_df["num_classes"].unique()), ordered=True)
    sns.scatterplot(data=sc_df, x="norm_ratio_z_s", y="norm_ratio_z_q",
                    hue="num_classes", palette=pal, ax=ax[1], alpha=0.3,
                    edgecolor="none")
    # NOTE: \| is mathtext-safe. With text.usetex=True you can restore
    # \lVert ... \rVert if you prefer the amsmath glyphs.
    ax[1].set_ylabel(rf"$\|{cfg['sym']}_\nu\| \,/\, \|{cfg['inset_tgt']}\|$",
                     fontsize=18)
    ax[1].set_xlabel(rf"$\|{cfg['sym']}_\nu\| \,/\, \|{cfg['tgt']}\|$",
                     fontsize=18)
    ax[1].legend(frameon=False, title="no. classes", title_fontsize=12,
                 loc=(1.05, 0.1))

    xs = np.linspace(0, lim, 200)
    ax[1].plot(xs, xs, color="0.5", lw=1.5, ls="--")
    if show_slope_guide:
        # F4: the slope varies with C, so a single global line is misleading.
        # Draw the guide for the LARGEST C (the visually dominant cloud).
        d1 = _one_sf(df)
        Cmax = int(d1["num_classes"].max())
        dm = d1[d1["num_classes"] == Cmax]
        m = float((dm["norm_ratio_z_q"] / dm["norm_ratio_z_s"]).mean())
        ax[1].plot(xs, m * xs, color="0.5", lw=1.5, ls="-.")
        ax[1].text(lim * 0.97, m * lim * 0.97,
                   f"slope {m:.2f}  ($C$={Cmax})  ",
                   fontsize=9, color="0.4", ha="right", va="bottom")
    ax[1].set_aspect("equal")
    ax[1].set_xlim(0, lim)
    ax[1].set_ylim(0, lim)

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=14)
    sns.despine(fig=fig)
    return fig, ax


def slope_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-C link gap: magnitude (slope) and direction (cosine). Report BOTH --
    slope alone cannot distinguish 'no bias' from 'right size, wrong direction'."""
    d = _one_sf(df)
    out = (d.assign(slope=d["norm_ratio_z_q"] / d["norm_ratio_z_s"])
             .groupby("num_classes")
             .agg(slope_mean=("slope", "mean"), slope_sd=("slope", "std")))
    ins = _dedup_inset(df).groupby("num_classes")["cos_sim_q_s"].agg(["mean", "std"])
    out["cos_mean"], out["cos_sd"] = ins["mean"], ins["std"]
    return out.reset_index()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import os
    import pickle

    UP = os.path.dirname(os.path.abspath(__file__))
    src = "/local_disk/vikrant/trident/logs"
    load = lambda pat: pd.DataFrame(
        pickle.load(open(glob.glob(os.path.join(src, pat))[0], "rb"))["data"])

    jobs = [
        ("*softmax_mc_analysis_gauss*",           "softmax",  "flare", "gauss"),
        ("*softmax_jacobian_mc_analysis_gauss*",  "jacobian", "flare", "gauss"),
        ("*softmax_mc_analysis_logistic*",        "softmax",  "crest", "logistic"),
        ("*softmax_jacobian_mc_analysis_logistic*", "jacobian", "crest", "logistic"),
    ]
    for pat, level, pal, noise in jobs:
        df = load(pat)
        # truncate logistic to the gaussian grid so the panels are comparable
        nu_max = 1024 if noise == "logistic" else None
        fig, _ = plot_level(df, level=level, palette=pal, nu_max=nu_max)
        fn = f"fig_{level}_{noise}_corrected.png"
        fig.savefig(os.path.join(UP, fn), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n=== {level.upper()} / {noise} -> {fn}")
        print(slope_table(df).to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))