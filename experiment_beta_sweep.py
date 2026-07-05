"""Run 4: Temperature sweep — how does PINN accuracy vary with β (inverse temperature)?
Higher β = higher barrier relative to kT = harder problem."""
import os, json, numpy as np, torch, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from experiment import MLP, train_pinn, evaluate, mc_committor, dw_grad, dw_A, dw_B, DEVICE, RESULTS, FIGURES
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(2); np.random.seed(2)
r = 0.25

gx = np.linspace(-1.85,1.85,15); gy = np.linspace(-1.1,1.1,15)
GX,GY = np.meshgrid(gx,gy)
test_pts = np.stack([GX.ravel(),GY.ravel()],1)
test_pts = test_pts[[not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in test_pts]]

x_int_np = np.random.uniform([-2,-1.2],[2,1.2],(6000,2))
mask = [not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in x_int_np]
x_int = torch.tensor(x_int_np[mask][:4000], dtype=torch.float32, device=DEVICE)
th = np.random.uniform(0,2*np.pi,400)
x_bcA = torch.tensor(np.stack([-1+r*np.cos(th),r*np.sin(th)],1), dtype=torch.float32, device=DEVICE)
x_bcB = torch.tensor(np.stack([ 1+r*np.cos(th),r*np.sin(th)],1), dtype=torch.float32, device=DEVICE)

results = {}
for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
    print(f"\nβ={beta} (barrier ≈ {1.0/beta:.2f} kT)...")
    q_mc = mc_committor(test_pts, dw_grad, beta, dt=5e-4, n_steps=8000, n_traj=150,
                        A_fn=dw_A, B_fn=dw_B)
    m = MLP().to(DEVICE)
    train_pinn(m, x_int, dw_grad, x_bcA, x_bcB, beta, epochs=2000)
    mae, rmse, _ = evaluate(m, test_pts, q_mc)
    results[beta] = {"mae": float(mae), "rmse": float(rmse), "q_range": [float(q_mc.min()), float(q_mc.max())]}
    print(f"  β={beta}: MAE={mae:.4f}")

fig, ax = plt.subplots(figsize=(7,4))
betas = sorted(results.keys())
ax.plot(betas, [results[b]["mae"] for b in betas], "o-", lw=2, color="tomato")
ax.set_xlabel("β (inverse temperature, mol/kcal)"); ax.set_ylabel("Committor MAE")
ax.set_title("PINN Accuracy vs. Temperature — Double-Well"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"beta_sweep.png"),dpi=150); plt.close()

with open(os.path.join(RESULTS,"beta_sweep.json"),"w") as f:
    json.dump({str(k): v for k,v in results.items()}, f, indent=2)
print("Beta sweep done.")
