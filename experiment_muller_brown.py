"""Run 5: Müller-Brown potential (third benchmark system, more complex landscape)."""
import os, json, numpy as np, torch, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from experiment import MLP, ResNet, train_pinn, train_flow, evaluate, mc_committor, DEVICE, RESULTS, FIGURES
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(3); np.random.seed(3)

# Müller-Brown potential (standard 3-basin system)
def mb_potential(x):
    A  = torch.tensor([-200.,-100.,-170., 15.], device=x.device)
    ax = torch.tensor([ -1.,  -1.,  -6.5,  0.7], device=x.device)
    bx = torch.tensor([  0.,   0.,  11.0,  0.6], device=x.device)
    cx = torch.tensor([-10., -10.,  -6.5,  0.7], device=x.device)
    x0 = torch.tensor([  1.,   0.,  -0.5, -1.0], device=x.device)
    y0 = torch.tensor([  0.,   0.5,  1.5,  1.0], device=x.device)
    V = torch.zeros(x.shape[0], device=x.device)
    for k in range(4):
        dx = x[:,0]-x0[k]; dy = x[:,1]-y0[k]
        V += A[k]*torch.exp(ax[k]*dx**2 + bx[k]*dx*dy + cx[k]*dy**2)
    return V / 20.0

def mb_grad(x):
    xc = x.clone().requires_grad_(True)
    mb_potential(xc).sum().backward()
    return xc.grad.detach()

# Basins: A near (-0.55, 1.44), B near (0.62, 0.03)
R_MB = 0.22
def mb_A(x,y): return (x+0.55)**2+(y-1.44)**2 < R_MB**2
def mb_B(x,y): return (x-0.62)**2+(y-0.03)**2 < R_MB**2

BETA = 3.0
DOMAIN = (-1.8, 1.2, -0.4, 2.2)

# Boundary points
th = np.random.uniform(0,2*np.pi,400)
bc_A_np = np.stack([-0.55+R_MB*np.cos(th), 1.44+R_MB*np.sin(th)],1)
bc_B_np = np.stack([ 0.62+R_MB*np.cos(th), 0.03+R_MB*np.sin(th)],1)
x_bcA = torch.tensor(bc_A_np, dtype=torch.float32, device=DEVICE)
x_bcB = torch.tensor(bc_B_np, dtype=torch.float32, device=DEVICE)

# Interior points
xl,xh,yl,yh = DOMAIN
x_int_np = np.random.uniform([xl,yl],[xh,yh],(8000,2))
mask = [not mb_A(p[0],p[1]) and not mb_B(p[0],p[1]) for p in x_int_np]
x_int_np = x_int_np[mask][:5000]
x_int = torch.tensor(x_int_np, dtype=torch.float32, device=DEVICE)
x_int_t = torch.tensor(x_int_np, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    V_int = mb_potential(x_int_t).cpu().numpy()
log_w = torch.tensor(-BETA*V_int, dtype=torch.float32, device=DEVICE)

# MC ground truth
gx = np.linspace(xl+0.1,xh-0.1,14); gy = np.linspace(yl+0.05,yh-0.05,14)
GX,GY = np.meshgrid(gx,gy)
test_pts = np.stack([GX.ravel(),GY.ravel()],1)
test_pts = test_pts[[not mb_A(p[0],p[1]) and not mb_B(p[0],p[1]) for p in test_pts]]
print(f"MC ground truth ({len(test_pts)} pts)...")
q_mc = mc_committor(test_pts, mb_grad, BETA, dt=5e-4, n_steps=12000, n_traj=200, A_fn=mb_A, B_fn=mb_B)
print(f"  q range: [{q_mc.min():.3f}, {q_mc.max():.3f}]")

results = {}
for name, Model in [("PINN-MLP", MLP), ("PINN-ResNet", ResNet)]:
    print(f"\n[{name}]")
    m = Model().to(DEVICE)
    t0 = time.time()
    train_pinn(m, x_int, mb_grad, x_bcA, x_bcB, BETA, epochs=4000)
    mae, rmse, qp = evaluate(m, test_pts, q_mc)
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  {time.time()-t0:.0f}s")
    results[name] = {"mae": float(mae), "rmse": float(rmse), "q_pred": qp.tolist()}

print("\n[FlowMatch-MLP]")
m = MLP().to(DEVICE)
t0 = time.time()
train_flow(m, x_int, log_w, bc_A_np, bc_B_np, x_bcA, x_bcB, epochs=4000)
mae, rmse, qp = evaluate(m, test_pts, q_mc)
print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  {time.time()-t0:.0f}s")
results["FlowMatch-MLP"] = {"mae": float(mae), "rmse": float(rmse), "q_pred": qp.tolist()}

# Plot
n_panels = len(results)+1
fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels,4))
kw = dict(cmap="RdBu_r", vmin=0, vmax=1, s=14)
sc = axes[0].scatter(test_pts[:,0], test_pts[:,1], c=q_mc, **kw)
axes[0].set_title("MC Ground Truth"); plt.colorbar(sc, ax=axes[0])
for ax, (k,v) in zip(axes[1:], results.items()):
    sc = ax.scatter(test_pts[:,0], test_pts[:,1], c=np.array(v["q_pred"]), **kw)
    ax.set_title(f"{k}\nMAE={v['mae']:.4f}"); plt.colorbar(sc, ax=ax)
fig.suptitle("Müller-Brown Committor", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"committor_muller_brown.png"),dpi=150,bbox_inches="tight"); plt.close()

results_save = {k: {kk:vv for kk,vv in v.items() if kk!="q_pred"} for k,v in results.items()}
results_save["q_mc"] = q_mc.tolist(); results_save["test_pts"] = test_pts.tolist()
with open(os.path.join(RESULTS,"muller_brown.json"),"w") as f: json.dump(results_save,f,indent=2)
print("Müller-Brown done.")
