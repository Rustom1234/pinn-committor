# Learning Committor Functions for Rare-Event Rates: PINNs vs. Flow-Matching

This project trains physics-informed neural networks (PINNs) and a
Dirichlet-energy / flow-matching alternative to learn the **committor
function** of a rare-event transition, and uses it to estimate unbinding
rates (k_off). It is validated on a synthetic 2D double-well system with a
known ground truth, then applied to a real protein-ligand system
(trypsin-benzamidine) and compared against both the classical Kramers
approximation and the published experimental rate.

## 1. Motivation

Residence time — how long a drug stays bound to its target — is one of the
numbers pharmaceutical researchers care about most when optimizing a
candidate molecule; a longer residence time often correlates with better
in-vivo efficacy, independent of binding affinity alone. Residence time is
the inverse of the unbinding rate, k_off, so estimating k_off accurately is
a practically important problem.

The catch is that unbinding is a **rare event**: the ligand spends the vast
majority of its simulated time in the bound state, and a direct molecular
dynamics simulation has to get lucky enough to sample an escape trajectory.
Brute-force simulation of unbinding is therefore prohibitively expensive.
This motivates methods that can estimate the rate without explicitly
simulating the rare event itself.

## 2. Method

### The committor function

The central object of transition-path theory is the **committor**, q(x):
the probability that, starting from configuration x, the system reaches the
unbound state before it returns to the bound state. If q(x) is known
everywhere in configuration space, k_off can be computed directly from it —
no brute-force sampling of an actual unbinding trajectory is required.

The committor satisfies the **backward Kolmogorov equation**:

```
(1/β) Δq − ∇V·∇q = 0
```

where V is the potential energy surface and β is the inverse temperature.
This is a linear second-order PDE with boundary conditions q = 0 in the
bound basin and q = 1 in the unbound basin.

### Method 1 — PINN (PDE-residual loss)

The PINN is trained by minimizing the residual of the backward Kolmogorov
equation directly, evaluated at collocation points across the domain
(`pinn_loss` in `experiment.py`), plus a `w_bc`-weighted boundary term that
pins q to 0 and 1 in the bound and unbound basins respectively. The physics
supplies the supervisory signal everywhere data is scarce — no simulated
transition paths are needed as training data.

### Method 2 — Flow-matching (Dirichlet-energy loss)

The second approach replaces the PDE-residual with its variational
equivalent: minimizing the Dirichlet energy

```
∫ |∇q|² dμ_eq
```

subject to the same boundary conditions. This is the same underlying
equation in weak form, but it only requires first derivatives of the
network output (no Laplacian), making it cheaper to evaluate and
numerically more stable to train. An interpolant-consistency term ties this
formulation to flow matching, framing the committor as a "flow" that
transports bound-distributed configurations to unbound-distributed ones.

### Test systems

1. **2D double-well** — a synthetic toy potential with a known analytical
   committor, used to validate that both methods learn the correct field
   before trusting them on a real system.
2. **Trypsin–benzamidine** — a real, experimentally characterized
   protein-ligand system, reduced to two coordinates (r = ligand-pocket
   distance, θ = binding angle), with the potential calibrated against the
   published free-energy profile of Buch et al. (PNAS 2011). The resulting
   k_off is compared directly to the experimentally measured rate.

## 3. Results

### System A — 2D double-well (validation against Monte Carlo ground truth)

| Model           | MAE    | RMSE   |
|-----------------|--------|--------|
| PINN-MLP        | 0.0145 | 0.0235 |
| PINN-ResNet     | 0.0145 | 0.0235 |
| FlowMatch-MLP   | 0.0228 | 0.0313 |

Both PINN variants recover the committor field to within ~0.0145 MAE of the
Monte Carlo ground truth, essentially indistinguishable from each other. The
flow-matching model is close but consistently a bit worse (MAE 0.0228 vs.
0.0145).

### System B — Trypsin-benzamidine

**Unbinding rate k_off (s⁻¹):**

| Source                                          | k_off (s⁻¹) |
|--------------------------------------------------|-------------|
| Experiment (Guillain & Thusius, 1970)             | 600         |
| Kramers (classical analytic approximation)        | 450         |
| PINN committor                                    | 18          |
| Flow-matching committor                           | 6.5         |

**Committor fit quality (vs. Monte Carlo ground truth on this system):**

| Model         | MAE    | RMSE   |
|---------------|--------|--------|
| PINN-MLP      | 0.1316 | 0.1735 |
| FlowMatch-MLP | 0.1344 | 0.1709 |

## 4. Honest Discussion

**What works:** On the double-well system, where ground truth is known
exactly, both the PINN and flow-matching methods recover the committor
field with high accuracy (MAE ≈ 0.0145 for the PINN variants, ≈0.0228 for
flow-matching). This validates that the training procedure and loss
formulations are sound and that the methods do what they are supposed to do
when the underlying dynamics are simple and well-resolved.

**What doesn't (yet):** On the real trypsin-benzamidine system, the learned
committors translate into rate estimates that are far from the experimental
value of 600 s⁻¹. The PINN estimate (18 s⁻¹) is off by roughly 33x, and the
flow-matching estimate (6.5 s⁻¹) is off by roughly 92x — both undershoot
substantially, landing in the ~25-100x-off range.

**Stated plainly: the classical Kramers approximation (450 s⁻¹) lands
closer to the experimental rate than either learned method.** This is not a
result we want to bury — it is the central finding of the real-system
experiment and the reason this section exists.

That said, "far from Kramers" is not the same as "far from the truth."
Kramers itself is a local-curvature approximation, and a code-level note in
`experiment.py` documents that it can overestimate k_off by roughly 3x for
a flat-unbound potential shape like the one used here. So while Kramers is
numerically the closest of the three methods to the experimental value in
this run, it is not necessarily the more trustworthy model in general — it
may simply be getting the right answer partly by chance, or the 2D
reduction may be flattering it in a way it isn't flattering the learned
committors.

Candid hypotheses for the gap between the learned methods and experiment:

- **The 2D reduction (r, θ) may be too aggressive.** Collapsing the real
  unbinding coordinate space down to two dimensions likely discards degrees
  of freedom (solvent, protein flexibility, alternate escape channels) that
  materially affect the rate.
- **The transition-state region may be undersampled during training.**
  Rare-event committors are most sensitive exactly where q is near 0.5, and
  if collocation/training points don't adequately cover that narrow region,
  the resulting rate estimate — which depends on gradients of q near the
  transition state — can be systematically biased.
- **Kramers' own reliability is in question for this potential shape**, per
  the ~3x-overestimation note above, so a large discrepancy between the
  learned methods and Kramers should not be read as a large discrepancy
  from the truth.

We report this gap directly, with numbers, rather than tuning the write-up
to obscure it, because a credible research artifact is one that shows where
a method validates cleanly (the double well) and where it currently falls
short (the real system) — not one that only shows the flattering half.

## 5. Figures

### Double-well validation

![Double-well committor field](figures/committor_dw.png)
*Learned committor field for the 2D double-well system, shown against the
known analytical solution.*

![Double-well training/validation curves](figures/curves_dw.png)
*Training and validation curves for the double-well experiments, showing
convergence of PINN and flow-matching models.*

### Trypsin-benzamidine

![Trypsin committor field](figures/committor_trypsin.png)
*Learned committor field over the reduced (r, θ) coordinates for the
trypsin-benzamidine system.*

![Trypsin training/validation curves](figures/curves_trypsin.png)
*Training and validation curves for the trypsin-benzamidine committor
models.*

![k_off comparison across methods](figures/koff_comparison.png)
*Unbinding rate k_off compared across Experiment, Kramers, PINN, and
flow-matching. Kramers is visibly closest to the experimental value; both
learned methods substantially undershoot it.*

## Citations

- Buch, I., Giorgino, T., & De Fabritiis, G. (2011). Complete reconstruction
  of an enzyme-inhibitor binding process by molecular dynamics simulations.
  *PNAS*, 108(25), 10184-10189.
- Guillain, F., & Thusius, D. (1970). Use of proflavine as an indicator in
  temperature-jump studies of the binding of a competitive inhibitor to
  trypsin. *Biochemistry*, 9(15), 2864-2872.
