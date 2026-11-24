#!/usr/bin/env python3
"""Process official ENSAE logo, topic cover photo, 4 minimalist icons, and Calibri charts."""

from __future__ import annotations

import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import cairosvg
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RAW_DIR = ASSETS_DIR / "raw"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRI_DIR = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
for fname in ("Calibri.ttf", "Calibrib.ttf", "Calibrii.ttf", "Calibriz.ttf"):
    fp = CALIBRI_DIR / fname
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))


def _fmt2(val: float) -> str:
    """Format float to 2 decimals using standard half-up rounding."""
    return str(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def process_ensae_logo(out_path: Path) -> None:
    """Trim whitespace around official ENSAE Paris / IP Paris PNG logo cleanly."""
    src = RAW_DIR / "LOGO-ENSAE.png"
    im = Image.open(src).convert("RGBA")
    arr = np.array(im)
    # Detect foreground pixels (non-transparent AND not pure white)
    alpha_mask = arr[:, :, 3] > 15
    non_white = (arr[:, :, 0] < 245) | (arr[:, :, 1] < 245) | (arr[:, :, 2] < 245)
    fg = alpha_mask & non_white
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Add 3% clean padding
    pad = int(0.03 * max(x1 - x0, y1 - y0))
    cropped = im.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(im.width, x1 + pad),
        min(im.height, y1 + pad),
    ))
    # Make near-white background transparent so it blends cleanly on slides
    carr = np.array(cropped)
    is_bg = (carr[:, :, 0] > 246) & (carr[:, :, 1] > 246) & (carr[:, :, 2] > 246)
    carr[is_bg, 3] = 0
    Image.fromarray(carr, mode="RGBA").save(out_path)


def process_cover_banner(out_path: Path) -> None:
    """Crop Paris La Defense financial skyline panorama cleanly to 2400x640 (3.75:1)."""
    src = RAW_DIR / "Panorama_La_Defense.jpg"
    im = Image.open(src).convert("RGB")
    # 2206 * 3.75 = 8272 px wide out of 13187 px; centering=(0.56, 0.50) captures
    # Grande Arche, central financial towers, and the Eiffel Tower in the distance.
    banner = ImageOps.fit(
        im,
        (2400, 640),
        method=Image.Resampling.LANCZOS,
        centering=(0.56, 0.50),
    )
    banner.save(out_path, quality=95)


def generate_icons() -> None:
    """Generate four minimalist navy line-art icons for Slide 4's 4-role architecture."""
    # Icon 1: Data & Empirical Replication (3 source feeds -> unified dataset)
    svg_data = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
  <g fill="none" stroke="#002752" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
    <rect x="24" y="42" width="66" height="52" rx="8"/>
    <line x1="38" y1="60" x2="76" y2="60"/>
    <line x1="38" y1="76" x2="66" y2="76"/>

    <rect x="117" y="42" width="66" height="52" rx="8"/>
    <line x1="131" y1="60" x2="169" y2="60"/>
    <line x1="131" y1="76" x2="159" y2="76"/>

    <rect x="210" y="42" width="66" height="52" rx="8"/>
    <line x1="224" y1="60" x2="262" y2="60"/>
    <line x1="224" y1="76" x2="252" y2="76"/>

    <path d="M 57 94 V 126 L 136 164"/>
    <path d="M 150 94 V 164"/>
    <path d="M 243 94 V 126 L 164 164"/>

    <rect x="64" y="168" width="172" height="90" rx="10"/>
    <line x1="64" y1="198" x2="236" y2="198"/>
    <line x1="86" y1="220" x2="214" y2="220"/>
    <line x1="86" y1="238" x2="188" y2="238"/>
  </g>
</svg>"""
    cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_data.png"))

    # Icon 2: Bootstrap Particle Filter & Dynamic Tracking
    svg_filter = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
  <g fill="none" stroke="#002752" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
    <rect x="34" y="54" width="204" height="148" rx="12"/>
    <polyline points="54,152 84,140 108,158 134,88 158,136 188,118 216,128" stroke="#008BD2"/>
    <circle cx="134" cy="112" r="4" fill="#002752"/>
    <circle cx="150" cy="116" r="4" fill="#002752"/>
    <circle cx="122" cy="126" r="4" fill="#002752"/>
    <circle cx="192" cy="186" r="48" fill="#F4F7FB" stroke="#002752"/>
    <polyline points="164,196 180,174 196,192 218,166" stroke="#008BD2"/>
    <line x1="228" y1="222" x2="264" y2="258" stroke-width="10"/>
  </g>
</svg>"""
    cairosvg.svg2png(bytestring=svg_filter.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_filter.png"))

    # Icon 3: Iterated Filtering (IF2): iterative cooling / swarm convergence
    svg_model = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
  <g fill="none" stroke="#002752" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M 58 64 H 40 V 214 H 58"/>
    <path d="M 164 64 H 182 V 214 H 164"/>
    <circle cx="72" cy="94" r="9" fill="#002752"/>
    <circle cx="111" cy="139" r="9" fill="#002752"/>
    <circle cx="150" cy="184" r="9" fill="#002752"/>
    <circle cx="72" cy="139" r="9" stroke="#008BD2"/>
    <circle cx="72" cy="184" r="9" stroke="#008BD2"/>
    <circle cx="111" cy="184" r="9" stroke="#008BD2"/>
    <line x1="105" y1="94" x2="117" y2="94"/>
    <line x1="144" y1="94" x2="156" y2="94"/>
    <line x1="144" y1="139" x2="156" y2="139"/>
    <line x1="196" y1="139" x2="222" y2="139"/>
    <polyline points="212,128 224,139 212,150"/>
    <rect x="230" y="96" width="48" height="86" rx="8"/>
    <line x1="242" y1="120" x2="266" y2="120" stroke="#008BD2"/>
    <line x1="242" y1="140" x2="266" y2="140"/>
    <line x1="242" y1="160" x2="266" y2="160" stroke="#008BD2"/>
    <circle cx="84" cy="252" r="10"/>
    <circle cx="150" cy="252" r="10"/>
    <circle cx="216" cy="252" r="10"/>
    <line x1="94" y1="252" x2="136" y2="252"/>
    <line x1="160" y1="252" x2="202" y2="252"/>
  </g>
</svg>"""
    cairosvg.svg2png(bytestring=svg_model.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_model.png"))

    # Icon 4: Bayesian PMCMC / SMC^2 posterior distribution & parameter inference
    svg_estimate = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
  <g fill="none" stroke="#002752" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
    <line x1="40" y1="224" x2="260" y2="224"/>
    <line x1="40" y1="74" x2="40" y2="224"/>
    <path d="M 52 214 C 95 210, 118 145, 150 145 C 182 145, 205 210, 248 214"
          stroke="#7A93AC" stroke-dasharray="10,9"/>
    <path d="M 64 218 C 110 218, 128 74, 150 74 C 172 74, 190 218, 236 218"
          stroke="#002752"/>
    <line x1="150" y1="74" x2="150" y2="224" stroke="#008BD2" stroke-dasharray="6,6"/>
    <circle cx="150" cy="74" r="8" fill="#008BD2" stroke="#002752"/>
    <circle cx="90" cy="256" r="8"/>
    <circle cx="150" cy="256" r="8" fill="#002752"/>
    <circle cx="210" cy="256" r="8"/>
    <line x1="98" y1="256" x2="142" y2="256"/>
    <line x1="158" y1="256" x2="202" y2="256"/>
  </g>
</svg>"""
    cairosvg.svg2png(bytestring=svg_estimate.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_estimate.png"))

    # Muted gray-slate variants for upcoming Phase III & IV placeholder cards on Slide 4
    svg_if2_muted = svg_model.replace("#002752", "#8A9BB0").replace("#008BD2", "#A3B3C6")
    cairosvg.svg2png(bytestring=svg_if2_muted.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_if2_muted.png"))

    svg_bayes_muted = svg_estimate.replace("#002752", "#8A9BB0").replace("#008BD2", "#A3B3C6").replace("#7A93AC", "#B0BECB")
    cairosvg.svg2png(bytestring=svg_bayes_muted.encode("utf-8"), write_to=str(ASSETS_DIR / "icon_bayes_muted.png"))


def render_equation(latex: str, out_path: Path, fontsize: float = 15.0, color: str = "#002752") -> None:
    """Render a crisp transparent high-DPI display equation PNG via Matplotlib mathtext."""
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig = plt.figure(figsize=(8.0, 0.9), dpi=400)
    fig.patch.set_alpha(0.0)
    fig.text(0.5, 0.5, latex, ha="center", va="center", fontsize=fontsize, color=color)
    fig.savefig(out_path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_equations() -> None:
    """Compile all presentation display formulas into crisp 400-DPI transparent PNGs."""
    render_equation(
        r"$\mathbf{y}_t = \mathbf{L}_t \mathbf{V}_t^{1/2}\mathbf{e}_t,\quad \mathbf{e}_t \sim \mathrm{iid}\ \mathcal{N}_p(\mathbf{0}, \mathbf{I}_p),\quad \Sigma_t = \mathbf{L}_t \mathbf{V}_t \mathbf{L}_t^\top$",
        ASSETS_DIR / "eq_chmsv_obs.png",
        fontsize=14.5,
        color="#002752",
    )
    render_equation(
        r"$h_{i,t} = \mu_i + \phi_i(h_{i,t-1} - \mu_i) + \eta_{i,t},\qquad q_{ij,t} = \mu_{ij} + \phi_{ij}(q_{ij,t-1} - \mu_{ij}) + \upsilon_{ij,t}$",
        ASSETS_DIR / "eq_chmsv_ar1.png",
        fontsize=14.0,
        color="#002752",
    )
    render_equation(
        r"$\mathrm{Stdev}(y_{i,t}\mid\mathcal{F}_{t-1}) = \sqrt{\Sigma_{t,ii}},\qquad \mathrm{Cor}(y_{i,t}, y_{j,t}\mid\mathcal{F}_{t-1}) = \frac{\Sigma_{t,ij}}{\sqrt{\Sigma_{t,ii}\,\Sigma_{t,jj}}}$",
        ASSETS_DIR / "eq_chmsv_corr.png",
        fontsize=14.0,
        color="#002752",
    )
    render_equation(
        r"$\mathrm{ESS}_t = \left(\sum_{n=1}^{N}(W_t^n)^2\right)^{-1} < \gamma N$",
        ASSETS_DIR / "eq_bpf_ess.png",
        fontsize=14.5,
        color="#002752",
    )
    render_equation(
        r"$\theta_t^{n} \sim \mathcal{N}\!\left(\theta_{t-1}^{n},\, \sigma_m^2 \mathbf{I}\right),\qquad \sigma_m = \sigma_0\,\alpha^{(m-1)/M}$",
        ASSETS_DIR / "eq_if2_cool.png",
        fontsize=14.5,
        color="#64748B",
    )
    render_equation(
        r"$p(\theta, \mathbf{x}_{0:T}\mid\mathbf{y}_{1:T}) \propto p(\theta)\,p(\mathbf{x}_0\mid\theta)\prod_{t=1}^{T} f_t(\mathbf{x}_t\mid\mathbf{x}_{t-1};\theta)\,g_t(\mathbf{y}_t\mid\mathbf{x}_t;\theta)$",
        ASSETS_DIR / "eq_bayes_post.png",
        fontsize=14.0,
        color="#64748B",
    )


def generate_slide5_stats_chart(out_path: Path) -> None:
    """Create crisp 2-panel Calibri bar chart comparing St. Dev. and Kurtosis across 4 assets."""
    assets = ["SPX", "Gold", "BTC", "ETH"]
    stdevs = [1.374, 0.926, 4.302, 5.693]
    kurtosis = [16.221, 6.528, 8.485, 8.553]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Calibri", "DejaVu Sans"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75), dpi=300)
    fig.patch.set_facecolor("white")

    x = np.arange(len(assets))
    w = 0.56

    colors_1 = ["#4E7496", "#4E7496", "#002752", "#002752"]
    bars1 = ax1.bar(x, stdevs, width=w, color=colors_1, edgecolor="none", zorder=3)
    ax1.set_title("Daily Return St. Dev. (%)", fontsize=10.5, fontweight="bold", color="#002752", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax1.set_ylim(0, 6.9)
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CBD5E1", alpha=0.8, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#94A3B8")
    ax1.spines["bottom"].set_color("#94A3B8")
    ax1.tick_params(axis="y", labelsize=9, colors="#475569")
    for b, val in zip(bars1, stdevs):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            val + 0.16,
            _fmt2(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#002752",
        )

    colors_2 = ["#002752", "#4E7496", "#4E7496", "#4E7496"]
    bars2 = ax2.bar(x, kurtosis, width=w, color=colors_2, edgecolor="none", zorder=3)
    ax2.set_title("Return Kurtosis (COVID Tail)", fontsize=10.5, fontweight="bold", color="#002752", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(assets, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_ylim(0, 19.5)
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CBD5E1", alpha=0.8, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#94A3B8")
    ax2.spines["bottom"].set_color("#94A3B8")
    ax2.tick_params(axis="y", labelsize=9, colors="#475569")
    for b, val in zip(bars2, kurtosis):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            val + 0.45,
            _fmt2(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#002752",
        )

    fig.tight_layout(pad=0.9)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def generate_slide10_clt_chart(out_path: Path) -> None:
    """Create CLT N^{-1/2} scaling bar + theoretical curve chart in Calibri."""
    n_vals = np.array([250, 500, 1000, 2000], dtype=float)
    std_emp = np.array([7.5588, 5.2768, 3.6621, 2.8856])
    std_theory = std_emp[0] * np.sqrt(n_vals[0] / n_vals)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Calibri", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(5.8, 2.75), dpi=300)
    fig.patch.set_facecolor("white")

    x = np.arange(len(n_vals))
    bars = ax.bar(
        x,
        std_emp,
        width=0.48,
        color="#002752",
        label="Empirical Std(log L) (10 runs)",
        zorder=3,
    )
    ax.plot(
        x,
        std_theory,
        color="#D97706",
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=2.0,
        label=r"Theoretical $\mathcal{O}(N^{-1/2})$ CLT Scaling",
        zorder=4,
    )

    ax.set_title(
        "Simulated 4-Asset SMC Variance Decay Across Particle Count N",
        fontsize=10.5,
        fontweight="bold",
        color="#002752",
        pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {int(n)}" for n in n_vals], fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Std of log L", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 9.6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CBD5E1", alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="y", labelsize=9, colors="#475569")

    for b, val, th in zip(bars, std_emp, std_theory):
        ax.text(
            b.get_x() + b.get_width() / 2,
            max(val, th) + 0.34,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#002752",
        )

    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    process_ensae_logo(ASSETS_DIR / "ensae_logo.png")
    process_cover_banner(ASSETS_DIR / "cover_banner.png")
    generate_icons()
    generate_equations()
    generate_slide5_stats_chart(ASSETS_DIR / "chart_slide5_stats.png")
    generate_slide10_clt_chart(ASSETS_DIR / "chart_slide10_clt.png")
    print(f"Generated ENSAE logo, cover banner, icons, equations, and Calibri charts in: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
