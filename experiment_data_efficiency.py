"""
Run 2: How few training points does PINN need vs Flow Matching?
Tests n_interior = [500, 1000, 2000, 4000, 8000] on the double-well.
"""
import os, json, numpy as np, torch, sys
sys.path.insert(0, os.path.dirname(__file__))
from experiment import (MLP, train_pinn, train_flow, evaluate, mc_committor,
                        dw_grad, dw_A, dw_B, DEVICE, RESULTS, FIGURES)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
BETA = 2.0
r = 0.25

# Fixed test set and MC ground truth
gx = np.linspace(-1.85, 1.85, 18); gy = np.linspace(-1.1, 1.1, 18)
GX, GY = np.meshgrid(gx, gy)
test_pts = np.stack([GX.ravel(), GY.ravel()], 1)
test_pts = test_pts[[not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in test_pts]]
print(f"Computing MC ground truth ({len(test_pts)} pts)...")
q_mc = mc_committor(test_pts, dw_grad, BETA, dt=1e-3, n_steps=5000, n_traj=150,
                    A_fn=dw_A, B_fn=dw_B)

th_A = np.random.uniform(0,2*np.pi,400)
bc_A = np.stack([-1+r*np.cos(th_A), r*np.sin(th_A)], 1)
th_B = np.random.uniform(0,2*np.pi,400)
bc_B = np.stack([ 1+r*np.cos(th_B), r*np.sin(th_B)], 1)
x_bcA = torch.tensor(bc_A, dtype=torch.float32, device=DEVICE)
x_bcB = torch.tensor(bc_B, dtype=torch.float32, device=DEVICE)

results = {"pinn": {}, "flow": {}}

for n in [500, 1000, 2000, 4000, 8000]:
    x_int_np = np.random.uniform([-2,-1.2],[2,1.2],(n*2,2))
    mask = [not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in x_int_np]
    x_int_np = x_int_np[mask][:n]
    x_int = torch.tensor(x_int_np, dtype=torch.float32, device=DEVICE)

    V_int = (x_int_np[:,0]**2-1)**2 + 2*x_int_np[:,1]**2
    log_w = torch.tensor(-BETA*V_int, dtype=torch.float32, device=DEVICE)

    # PINN
    m = MLP().to(DEVICE)
    train_pinn(m, x_int, dw_grad, x_bcA, x_bcB, BETA, epochs=2000)
    mae, rmse, _ = evaluate(m, test_pts, q_mc)
    results["pinn"][n] = {"mae": float(mae), "rmse": float(rmse)}
    print(f"  PINN  n={n:5d}: MAE={mae:.4f}")

    # Flow
    m = MLP().to(DEVICE)
    train_flow(m, x_int, log_w, bc_A, bc_B, x_bcA, x_bcB, epochs=2000)
    mae, rmse, _ = evaluate(m, test_pts, q_mc)
    results["flow"][n] = {"mae": float(mae), "rmse": float(rmse)}
    print(f"  Flow  n={n:5d}: MAE={mae:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(7,4))
ns = sorted(results["pinn"].keys())
ax.plot(ns, [results["pinn"][n]["mae"] for n in ns], "o-", label="PINN", lw=2)
ax.plot(ns, [results["flow"][n]["mae"] for n in ns], "s--", label="Flow Matching", lw=2)
ax.set_xlabel("Interior collocation points"); ax.set_ylabel("Committor MAE")
ax.set_title("Data Efficiency: PINN vs Flow Matching"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "data_efficiency.png"), dpi=150)
plt.close()

with open(os.path.join(RESULTS, "data_efficiency.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Data efficiency done. Results saved.")
