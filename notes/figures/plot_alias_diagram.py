"""2D diagram of the alias pairing under 2:1 subsampling.

The fine-mode region is the square [-k_Ny,2N, k_Ny,2N]^2.  The coarse zone
[-k_Ny,N, k_Ny,N]^2 sits in the center.  Each coarse mode k_c has four
contributors: the principal (n = (0,0)) at k_c itself, plus three aliases
at k_c + π n / Δx_f for n ∈ {0,1}^2, n ≠ (0,0), which (after periodic
identification) sit in the three other quadrants of the fine zone.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Use units where k_Ny,2N = 1, so k_Ny,N = 1/2.
kNy_2N = 1.0
kNy_N  = 0.5

fig, ax = plt.subplots(figsize=(7, 7))

# Fine-mode region (light red wings) and coarse zone (light blue center).
ax.add_patch(Rectangle((-kNy_2N, -kNy_2N), 2*kNy_2N, 2*kNy_2N,
                       facecolor='red', alpha=0.10, ec='darkred', lw=1.2))
ax.add_patch(Rectangle((-kNy_N, -kNy_N), 2*kNy_N, 2*kNy_N,
                       facecolor='blue', alpha=0.12, ec='navy', lw=1.4))

# Axes through origin.
ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)

# Pick a representative coarse mode k_c inside the coarse zone and show its
# four partners under (n_x, n_y) ∈ {0,1}^2.
k_c = np.array([0.18, 0.13])    # principal location

# Compute the four partners using periodic identification: shift by k_Ny,2N
# along an axis means jumping to the opposite-sign outer wing along that axis.
def alias_partner(k_c, n):
    out = k_c.copy()
    for axis in (0, 1):
        if n[axis] == 1:
            # shift by +k_Ny,2N, then wrap into the principal period
            out[axis] = k_c[axis] - kNy_2N if k_c[axis] >= 0 else k_c[axis] + kNy_2N
    return out

n_vecs   = [(0,0), (1,0), (0,1), (1,1)]
labels   = ['principal n=(0,0)',
            'alias n=(1,0)',
            'alias n=(0,1)',
            'alias n=(1,1)']
colors   = ['C0', 'C1', 'C2', 'C3']

partners = [alias_partner(k_c, n) for n in n_vecs]

# Mark each partner.
for p, n, lab, col in zip(partners, n_vecs, labels, colors):
    ax.plot(p[0], p[1], 'o', color=col, ms=10, zorder=4,
            label=f'{lab}  at ({p[0]:+.2f}, {p[1]:+.2f})')

# Arrows from each alias to the principal (showing folding).
for p, col in zip(partners[1:], colors[1:]):
    arr = FancyArrowPatch((p[0], p[1]), (k_c[0], k_c[1]),
                          arrowstyle='->', mutation_scale=14,
                          color=col, lw=1.2,
                          connectionstyle='arc3,rad=0.18')
    ax.add_patch(arr)

# Tick labels.
for x, lbl in [(-kNy_2N, r'$-k_{\rm Ny,2N}$'),
               (-kNy_N,  r'$-k_{\rm Ny,N}$'),
               ( kNy_N,  r'$+k_{\rm Ny,N}$'),
               ( kNy_2N, r'$+k_{\rm Ny,2N}$')]:
    ax.plot([x, x], [-kNy_2N - 0.02, -kNy_2N - 0.06], 'k-', lw=1)
    ax.text(x, -kNy_2N - 0.10, lbl, ha='center', va='top', fontsize=9)
for y, lbl in [(-kNy_2N, r'$-k_{\rm Ny,2N}$'),
               (-kNy_N,  r'$-k_{\rm Ny,N}$'),
               ( kNy_N,  r'$+k_{\rm Ny,N}$'),
               ( kNy_2N, r'$+k_{\rm Ny,2N}$')]:
    ax.plot([-kNy_2N - 0.02, -kNy_2N - 0.06], [y, y], 'k-', lw=1)
    ax.text(-kNy_2N - 0.10, y, lbl, ha='right', va='center', fontsize=9)

# Region labels.
ax.text(0, 0, 'coarse zone', ha='center', va='center',
        color='navy', fontsize=10, alpha=0.5)
ax.text(-0.75, 0.75, 'outer\nwing', ha='center', va='center',
        color='darkred', fontsize=9, alpha=0.6)
ax.text( 0.75, 0.75, 'outer\nwing', ha='center', va='center',
        color='darkred', fontsize=9, alpha=0.6)
ax.text(-0.75,-0.75, 'outer\nwing', ha='center', va='center',
        color='darkred', fontsize=9, alpha=0.6)
ax.text( 0.75,-0.75, 'outer\nwing', ha='center', va='center',
        color='darkred', fontsize=9, alpha=0.6)

ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-1.35, 1.55)
ax.set_ylim(-1.30, 1.20)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax.set_title('Alias pairing under $2{:}1$ subsampling (2D)', fontsize=11)
plt.tight_layout()
plt.savefig('alias_diagram.pdf')
print("wrote alias_diagram.pdf")
