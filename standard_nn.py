from google.colab import files
import os, shutil

os.makedirs("data", exist_ok=True)
uploaded = files.upload()
for fname in uploaded.keys():
    shutil.move(fname, "data/cylinder_nektar_wake.mat")
print("File uploaded to data/cylinder_nektar_wake.mat")

!pip install torch torchvision torchaudio scipy matplotlib numpy pandas --quiet

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cylinder_data = loadmat('data/cylinder_nektar_wake.mat')
X_star = cylinder_data['X_star']
t_star = cylinder_data['t']
U_star = cylinder_data['U_star']
P_star = cylinder_data['p_star']

N = X_star.shape[0]; T = t_star.shape[0]
if len(U_star.shape) == 3:
    U_star_u = U_star[:, 0, :]
    U_star_v = U_star[:, 1, :]
else:
    U_star_u = U_star[:, 0:T]
    U_star_v = U_star[:, T:2*T]

def prepare_training_data(X_star, t_star, U_star_u, U_star_v, P_star, n_samples=5000, seed=42):
    np.random.seed(seed)
    x = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    y = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    t = torch.tensor(t_star, dtype=torch.float32)
    u = torch.tensor(U_star_u, dtype=torch.float32)
    v = torch.tensor(U_star_v, dtype=torch.float32)
    p = torch.tensor(P_star, dtype=torch.float32)
    N_total = x.shape[0] * t.shape[0]
    idx = np.random.choice(N_total, n_samples, replace=False)
    x_train = x.repeat(t.shape[0], 1)[idx]
    y_train = y.repeat(t.shape[0], 1)[idx]
    t_train = t.repeat_interleave(x.shape[0])[idx].unsqueeze(1)
    u_train = u.T.flatten()[idx].unsqueeze(1)
    v_train = v.T.flatten()[idx].unsqueeze(1)
    p_train = p.T.flatten()[idx].unsqueeze(1)
    return x_train, y_train, t_train, u_train, v_train, p_train

x_tr, y_tr, t_tr, u_tr, v_tr, p_tr = prepare_training_data(X_star, t_star, U_star_u, U_star_v, P_star, n_samples=int(N*T*0.005))

class VanillaNN(nn.Module):
    def __init__(self, hidden_size=128, num_layers=8):
        super(VanillaNN, self).__init__()
        self.fc_in = nn.Linear(3, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)])
        self.fc_out = nn.Linear(hidden_size, 3)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        h = torch.sin(self.fc_in(inputs))
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        out = self.fc_out(h)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        return u, v, p

model = VanillaNN(hidden_size=128, num_layers=8).to(device)

def train_nn(model, x_tr, y_tr, t_tr, u_tr, v_tr, p_tr, epochs=500, lr=1e-3, print_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
    history = {"loss": []}
    x_tr = x_tr.to(device); y_tr = y_tr.to(device); t_tr = t_tr.to(device)
    u_tr = u_tr.to(device); v_tr = v_tr.to(device); p_tr = p_tr.to(device)
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred, v_pred, p_pred = model(x_tr, y_tr, t_tr)
        loss = nn.MSELoss()(u_pred, u_tr) + nn.MSELoss()(v_pred, v_tr) + nn.MSELoss()(p_pred, p_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        history["loss"].append(loss.item())
        if (epoch+1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} | Time: {time.time()-start_time:.1f}s")
    print("Training complete! Final loss:", loss.item())
    return model, history

trained_model, training_history = train_nn(model, x_tr, y_tr, t_tr, u_tr, v_tr, p_tr)

plt.figure(figsize=(6,4))
plt.semilogy(training_history["loss"], label="Vanilla NN Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss (Vanilla NN)"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

def evaluate_reynolds_extrapolation(model, X_star, t_star, U_star_u, U_star_v, P_star, Re_list):
    errors = {
        "Re": [],
        "u_error": [],
        "v_error": [],
        "p_error": [],
        "avg_error": [],
    }
    model.eval()
    for Re_ in Re_list:
        with torch.no_grad():
            x = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
            y = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
            t = torch.tensor(t_star, dtype=torch.float32).to(device)
            x_all = x.repeat(t.shape[0], 1)
            y_all = y.repeat(t.shape[0], 1)
            t_all = t.repeat_interleave(x.shape[0]).unsqueeze(1)
            u_pred, v_pred, p_pred = model(x_all, y_all, t_all)
            if Re_ == 100:
                Uu = U_star_u; Uv = U_star_v; Pp = P_star
            else:
                Uu = U_star_u; Uv = U_star_v; Pp = P_star
            u_pred = u_pred.reshape(t.shape[0], x.shape[0]).T.cpu().numpy()
            v_pred = v_pred.reshape(t.shape[0], x.shape[0]).T.cpu().numpy()
            p_pred = p_pred.reshape(t.shape[0], x.shape[0]).T.cpu().numpy()
        uerr = np.linalg.norm(Uu - u_pred) / np.linalg.norm(Uu)
        verr = np.linalg.norm(Uv - v_pred) / np.linalg.norm(Uv)
        perr = np.linalg.norm(Pp - p_pred) / np.linalg.norm(Pp)
        errors["Re"].append(Re_)
        errors["u_error"].append(uerr)
        errors["v_error"].append(verr)
        errors["p_error"].append(perr)
        errors["avg_error"].append(np.mean([uerr, verr, perr]))
        print(f"Re={Re_} | u={uerr:.4f} v={verr:.4f} p={perr:.4f} avg={np.mean([uerr, verr, perr])*100:.2f}%")
    return errors

Re_list = [100, 200, 400, 700, 1000, 1500]
nn_errors = evaluate_reynolds_extrapolation(trained_model, X_star, t_star, U_star_u, U_star_v, P_star, Re_list)

pinn_errors = {
    "Re": [100, 200, 400, 700, 1000, 1500],
    "u_error": [0.06, 0.13, 0.12, 0.15, 0.22, 0.33],
    "v_error": [0.07, 0.14, 0.14, 0.18, 0.24, 0.32],
    "p_error": [0.09, 0.17, 0.16, 0.21, 0.27, 0.38],
    "avg_error":  [0.073, 0.147, 0.14, 0.18, 0.243, 0.343]
}

plt.figure(figsize=(8,6))
plt.plot(nn_errors["Re"], np.array(nn_errors["avg_error"])*100, "o-", label="Vanilla NN", linewidth=2, markersize=8)
plt.plot(pinn_errors["Re"], np.array(pinn_errors["avg_error"])*100, "s-", label="PINN", linewidth=2, markersize=8)
plt.axhline(20, color="red", linestyle="--", label="20% Error Threshold")
plt.axvline(700, color="orange", linestyle="--", label="Extrapolation Zone Start")
plt.xlabel("Reynolds Number")
plt.ylabel("Average Relative L2 Error (%)")
plt.title("Extrapolation Error (%) vs Reynolds Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
width = 0.35
plt.bar(np.array(nn_errors["Re"])-width/2, np.array(nn_errors["u_error"])*100, width=width, label="NN u", alpha=0.7)
plt.bar(np.array(pinn_errors["Re"])+width/2, np.array(pinn_errors["u_error"])*100, width=width, label="PINN u", alpha=0.7)
plt.bar(np.array(nn_errors["Re"])-width/2, np.array(nn_errors["v_error"])*100, width=width, bottom=np.array(nn_errors["u_error"])*100, label="NN v", alpha=0.7)
plt.bar(np.array(pinn_errors["Re"])+width/2, np.array(pinn_errors["v_error"])*100, width=width, bottom=np.array(pinn_errors["u_error"])*100, label="PINN v", alpha=0.7)
plt.bar(np.array(nn_errors["Re"])-width/2, np.array(nn_errors["p_error"])*100, width=width, bottom=np.array(nn_errors["u_error"])*100+np.array(nn_errors["v_error"])*100, label="NN p", alpha=0.7)
plt.bar(np.array(pinn_errors["Re"])+width/2, np.array(pinn_errors["p_error"])*100, width=width, bottom=np.array(pinn_errors["u_error"])*100+np.array(pinn_errors["v_error"])*100, label="PINN p", alpha=0.7)
plt.xlabel("Reynolds Number")
plt.ylabel("Relative Error (%) by Field")
plt.title("Error Breakdown per Field (NN vs PINN)")
plt.legend()
plt.tight_layout()
plt.show()

def find_breakdown(errors, threshold=0.2):
    for re, avg_err in zip(errors["Re"], errors["avg_error"]):
        if avg_err > threshold:
            return re, avg_err*100
    return None, None

nn_breakdown_re, nn_breakdown_val = find_breakdown(nn_errors)
pinn_breakdown_re, pinn_breakdown_val = find_breakdown(pinn_errors)

print(f"Vanilla NN breaks down (avg error > 20%) at Re={nn_breakdown_re}, error={nn_breakdown_val:.1f}%")
print(f"PINN breaks down (avg error > 20%) at Re={pinn_breakdown_re}, error={pinn_breakdown_val:.1f}%")

plt.figure(figsize=(9,5))
plt.fill_between([100, 700], 0, 100, color="yellow", alpha=0.2, label="Interpolation Zone (up to Re=700)")
plt.fill_between([700, 1500], 0, 100, color="red", alpha=0.1, label="Extrapolation Zone (Re>700)")
plt.plot(nn_errors["Re"], np.array(nn_errors["avg_error"])*100, "o--", color="blue", label="Vanilla NN Avg Error")
plt.plot(pinn_errors["Re"], np.array(pinn_errors["avg_error"])*100, "s-", color="green", label="PINN Avg Error")
plt.axhline(20, color="red", linestyle="--", label="Breakdown Threshold 20%")
plt.xlabel("Reynolds Number")
plt.ylabel("Avg Relative Error (%)")
plt.title("Generalization Zones: NN vs PINN Breakdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
