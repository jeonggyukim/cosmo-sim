# Figures to implement

## Status legend
- [ ] pending
- [x] done

---

## fft_review.tex

- [x] **butterfly.tikz** — N=8 signal flow graph (inline TikZ in §2.1)
  - 4 columns × 8 nodes; connections color-coded by stage (3 stages)
  - Shows O(N log N): N/2=4 butterflies per stage, log₂N=3 stages
  - Single butterfly detail callout on the side

## lpt_review.tex

- [x] **pancake.pdf** — Zel'dovich pancake collapse (`notes/plot_pancake.py`)
  - Left: density 1+δ = 1/|1−D₊cosq| vs Eulerian x, for D₊=0.3,0.6,0.9,1.0,1.2
  - Right: particle trajectories x(q)=q−D₊sinq showing fold-over at shell crossing
  - Placed in §4.4 (Shell crossing and the Zel'dovich pancake)

- [x] **growth.pdf** — Growth factor and growth rate (`notes/plot_growth.py`)
  - Left: D₊(z) normalized to 1 at z=0; flat ΛCDM (Ωm=0.3, ΩΛ=0.7)
  - Right: f(z) = Ωm(a)^0.55; also f^(2)(z) = 2 Ωm(a)^(4/7)
  - Dashed reference lines: matter-dom limit D₊∝1/(1+z), f→1, f^(2)→2
  - Placed after eq:fgrowth in §2.2

- [x] **icgen_flow.tikz** — IC generation flowchart (inline TikZ in §6)
  - Boxes: draw δ → φ^(1) → IFFTs → Ψ^(1); branch for 2LPT source → φ^(2) → Ψ^(2)
  - Final: x = q + D₊Ψ^(1) + D₊^(2)Ψ^(2); v = ℋ(...)
  - Placed at end of §6 as a summary figure

---

## Makefile additions needed

```
pancake.pdf : plot_pancake.py
    $(PYTHON) $<

growth.pdf : plot_growth.py
    $(PYTHON) $<
```

Add `pancake.pdf growth.pdf` to FIGURES variable.
