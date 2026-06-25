"""Static isometric schematic of the Monash Tri-HB system for the paper.
Ports the geometry of the Streamlit Overview animation to matplotlib.
Run:  py -3.13 scripts/export_paper_schematic.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C30, S30 = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
r, c = 0.12, 0.17
rc = r * 1.5

GRAY = ["#B4B2A9", "#6d6c66", "#8d8c84"]
DGRAY = ["#9a9890", "#54534e", "#6e6d67"]
AMBER = ["#FAC775", "#BA7517", "#EF9F27"]
BLUE = ["#85B7EB", "#185FA5", "#378ADD"]
CORAL = ["#F0997B", "#993C1D", "#D85A30"]
EDGE = "#3f3e3c"


def P(x, y, z):
    return ((x - z) * C30, y - (x + z) * S30)


def box(ax, x0, y0, z0, x1, y1, z1, cols, alpha=1.0):
    top = [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)]
    lz = [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]
    rx = [P(x1, y0, z0), P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0)]
    for pts, col in ((lz, cols[1]), (rx, cols[2]), (top, cols[0])):
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor=EDGE,
                             linewidth=0.6, joinstyle="round", alpha=alpha, zorder=2))


def main():
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    solids = []

    def add(x0, y0, z0, x1, y1, z1, cols):
        solids.append((x0 + x1 + y0 + y1 + z0 + z1, (x0, y0, z0, x1, y1, z1, cols)))

    # striker + X train
    add(-2.95, -r, -r, -2.59, r, r, CORAL)
    add(-2.25, -r, -r, -c, r, r, GRAY)            # incident bar
    add(c, -r, -r, 1.75, r, r, GRAY)              # transmission bar
    add(1.75, -r * 0.8, -r * 0.8, 2.05, r * 0.8, r * 0.8, DGRAY)  # absorption
    add(2.05, -rc, -rc, 2.40, rc, rc, BLUE)       # X hydraulic load cylinder
    # Y bars + cylinders
    add(-r, c, -r, r, 1.25, r, GRAY)
    add(-r, -1.25, -r, r, -c, r, GRAY)
    add(-rc, 1.25, -rc, rc, 1.55, rc, BLUE)
    add(-rc, -1.55, -rc, rc, -1.25, rc, BLUE)
    # Z bars + cylinders
    add(-r, -r, c, r, r, 1.25, GRAY)
    add(-r, -r, -1.25, r, r, -c, GRAY)
    add(-rc, -rc, 1.25, rc, rc, 1.55, BLUE)
    add(-rc, -rc, -1.55, rc, rc, -1.25, BLUE)
    # specimen
    add(-c, -c, -c, c, c, c, AMBER)

    for _, args in sorted(solids, key=lambda s: s[0]):
        box(ax, *args)

    # axis triad
    o = np.array(P(-2.7, -1.0, -1.0))
    for d, lab in (((1, 0, 0), "X"), ((0, 1, 0), "Y"), ((0, 0, 1), "Z")):
        p1 = np.array(P(*[d[k] * 0.55 - 2.7 * (k == 0) - 1.0 * (k == 1) - 1.0 * (k == 2) for k in range(3)]))
        tip = np.array(P(-2.7 + d[0] * 0.55, -1.0 + d[1] * 0.55, -1.0 + d[2] * 0.55))
        ax.annotate("", xy=tip, xytext=o, arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
        ax.text(*(tip + (tip - o) * 0.18), lab, fontsize=12, ha="center", va="center", color="#222")

    def lbl(x, y, z, text, dx=0.0, dy=0.0, ha="center"):
        px, py = P(x, y, z)
        ax.annotate(text, xy=(px, py), xytext=(px + dx, py + dy), fontsize=10.5,
                    ha=ha, va="center", color="#1a1a1a",
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7))

    lbl(-2.77, 0.0, 0.0, "Striker\n(gas gun)", dx=-0.55, dy=0.5, ha="right")
    lbl(-1.4, r, 0.0, "Incident bar (X)", dx=-0.2, dy=0.9)
    lbl(0.95, r, 0.0, "Transmission bar (X)", dx=0.4, dy=0.95)
    lbl(2.2, rc, 0.0, "Absorption + X\nhydraulic load", dx=0.55, dy=0.4, ha="left")
    lbl(0.0, 1.4, 0.0, "Y output bars +\nhydraulic confinement", dx=0.2, dy=0.55)
    lbl(0.0, 0.0, 1.35, "Z output bars +\nhydraulic confinement", dx=-0.65, dy=0.5, ha="right")
    lbl(c, c, c, "Cubic specimen\n(6 interfaces)", dx=0.7, dy=-0.7, ha="left")

    ax.autoscale_view()
    ax.margins(0.16)
    fig.tight_layout()
    fig.savefig(OUT / "trihb_system_schematic.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT / 'trihb_system_schematic.png'}")


if __name__ == "__main__":
    main()
