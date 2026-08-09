#!/usr/bin/env python3
"""
plot_seed_dice.py — the dice illustration of what a seed search does.

Two fair dice. Keeping only the rolls that satisfy a condition on the sum
changes quantities that the condition never mentioned. Everything here is
exact: 36 equally likely outcomes, counted.

  keep sum = 7   the first die alone is untouched (mean 3.5, spread 1.71),
                 but d1 + 2 d2 has its spread cut from 3.82 to 1.71
  keep sum >= 10 the first die alone moves: mean 3.5 -> 5.33,
                 spread 1.71 -> 0.75

Figure for notes/seed_selection.tex.

Run:
    make -C notes figures
or, from this directory:
    conda run -n cosmo python plot_seed_dice.py

Writes seed_dice.pdf and seed_dice.png next to this script.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAIR = "0.75"
KEPT = "C3"


def moments(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), v.std()


def main():
    d1, d2 = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
    d1, d2 = d1.ravel(), d2.ravel()

    seven = (d1 + d2) == 7
    ten = (d1 + d2) >= 10

    combo = d1 + 2 * d2
    m_all, s_all = moments(combo)
    m_7, s_7 = moments(combo[seven])
    m1_all, s1_all = moments(d1)
    m1_10, s1_10 = moments(d1[ten])

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    a = ax[0]
    a.scatter(d1[~seven], d2[~seven], s=210, color=FAIR, edgecolor="none",
              zorder=2)
    a.scatter(d1[seven], d2[seven], s=210, color=KEPT, edgecolor="k",
              linewidth=0.6, zorder=3)
    a.set_xticks(range(1, 7))
    a.set_yticks(range(1, 7))
    a.set_xlim(0.4, 6.6)
    a.set_ylim(0.4, 6.6)
    a.set_xlabel("first die $X_1$")
    a.set_ylabel("second die $X_2$")
    a.set_title("(a) keep only the rolls with $X_1 + X_2 = 7$", fontsize=11)
    a.text(0.5, 6.25, "6 of 36 rolls survive", fontsize=10, color=KEPT)
    a.set_aspect("equal")

    a = ax[1]
    bins = np.arange(2.5, 19.5, 1.0)
    a.hist(combo, bins=bins, color=FAIR, label="all 36 rolls")
    a.hist(combo[seven], bins=bins, color=KEPT, alpha=0.85,
           label="rolls with sum 7")
    a.set_ylim(0, 4.4)
    a.set_xlabel("$Y = X_1 + 2X_2$")
    a.set_ylabel("number of rolls")
    a.set_title("(b) $Y = X_1 + 2X_2$, which the rule never mentioned",
                fontsize=11)
    a.text(0.03, 0.80,
           f"spread {s_all:.2f}  $\\rightarrow$  {s_7:.2f}\n"
           f"same average ({m_all:.1f} $\\rightarrow$ {m_7:.1f})",
           transform=a.transAxes, fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="upper right")

    a = ax[2]
    bins = np.arange(0.5, 7.5, 1.0)
    a.hist(d1, bins=bins, color=FAIR, label="all 36 rolls")
    a.hist(d1[ten], bins=bins, color=KEPT, alpha=0.85,
           label="rolls with sum $\\geq$ 10")
    a.set_xticks(range(1, 7))
    a.set_ylim(0, 8.8)
    a.set_xlabel("first die $X_1$")
    a.set_ylabel("number of rolls")
    a.set_title("(c) keeping sum $\\geq$ 10 moves the first die as well",
                fontsize=11)
    a.text(0.03, 0.80,
           f"average {m1_all:.2f}  $\\rightarrow$  {m1_10:.2f}\n"
           f"spread {s1_all:.2f}  $\\rightarrow$  {s1_10:.2f}",
           transform=a.transAxes, fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="upper right")

    for a in ax:
        a.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Selecting on the sum $X_1+X_2$ changes quantities the rule never mentioned",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig("seed_dice.pdf", bbox_inches="tight")
    fig.savefig("seed_dice.png", dpi=150, bbox_inches="tight")

    print(f"first die + twice second: spread {s_all:.3f} -> {s_7:.3f} "
          f"(mean {m_all:.2f} -> {m_7:.2f})")
    print(f"first die, sum >= 10:     spread {s1_all:.3f} -> {s1_10:.3f} "
          f"(mean {m1_all:.2f} -> {m1_10:.2f})")
    print("Saved: seed_dice.pdf, seed_dice.png")


if __name__ == "__main__":
    main()
