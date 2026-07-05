"""Run 3: Architecture ablation — width and depth on double-well."""
import os, json, numpy as np, torch, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from experiment import train_pinn, evaluate, mc_committor, dw_grad, dw_A, dw_B, DEVICE, RESULTS, FIGURES
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(1); np.random.seed(1)
BETA = 2.0; r = 0.25

class FlexMLP(nn.Module):
    def __init__(self, hidden, depth):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(hidden,hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

gx = np.linspace(-1.85,1.85,18); gy = np.linspace(-1.1,1.1,18)
GX,GY = np.meshgrid(gx,gy)
test_pts = np.stack([GX.ravel(),GY.ravel()],1)
test_pts = test_pts[[not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in test_pts]]
print(f"MC ground truth ({len(test_pts)} pts)...")
q_mc = mc_committor(test_pts, dw_grad, BETA, dt=1e-3, n_steps=5000, n_traj=150, A_fn=dw_A, B_fn=dw_B)

x_int_np = np.random.uniform([-2,-1.2],[2,1.2],(6000,2))
mask = [not dw_A(p[0],p[1]) and not dw_B(p[0],p[1]) for p in x_int_np]
x_int = torch.tensor(x_int_np[mask][:4000], dtype=torch.float32, device=DEVICE)
th = np.random.uniform(0,2*np.pi,400)
x_bcA = torch.tensor(np.stack([-1+r*np.cos(th),r*np.sin(th)],1), dtype=torch.float32, device=DEVICE)
x_bcB = torch.tensor(np.stack([ 1+r*np.cos(th),r*np.sin(th)],1), dtype=torch.float32, device=DEVICE)

results = {}
for depth in [2, 3, 4, 6]:
    for hidden in [64, 128, 256]:
        name = f"d{depth}_h{hidden}"
        m = FlexMLP(hidden, depth).to(DEVICE)
        params = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        train_pinn(m, x_int, dw_grad, x_bcA, x_bcB, BETA, epochs=2000)
        mae, rmse, _ = evaluate(m, test_pts, q_mc)
        results[name] = {"mae": float(mae), "rmse": float(rmse),
                         "params": params, "time_s": float(time.time()-t0)}
        print(f"  {name}: MAE={mae:.4f}  params={params}")

fig, ax = plt.subplots(figsize=(8,5))
for depth in [2,3,4,6]:
    hs = [64,128,256]
    maes = [results[f"d{depth}_h{h}"]["mae"] for h in hs]
    ax.plot(hs, maes, "o-", label=f"depth={depth}")
ax.set_xlabel("Hidden width"); ax.set_ylabel("Committor MAE")
ax.set_title("PINN Architecture Ablation — Double-Well"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"arch_ablation.png"),dpi=150); plt.close()

with open(os.path.join(RESULTS,"arch_ablation.json"),"w") as f: json.dump(results,f,indent=2)
print("Arch ablation done.")
