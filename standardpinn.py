# PINN for Reynolds Number Extrapolation
# Project: Quantifying Generalization Limits of PINNs

# Install dependencies
!pip install -q torch torchvision torchaudio scipy matplotlib numpy pandas

# Clone repository
import os
if not os.path.exists('PINN_navier_stokes'):
    !git clone https://github.com/Matt2371/PINN_navier_stokes.git
os.chdir('/content/PINN_navier_stokes')

# Download dataset
print("Downloading cylinder_nektar_wake.mat dataset...")
if not os.path.exists('data/cylinder_nektar_wake.mat'):
    !wget -q --no-check-certificate \
        'https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat' \
        -O data/cylinder_nektar_wake.mat

if os.path.exists('data/cylinder_nektar_wake.mat'):
    file_size = os.path.getsize('data/cylinder_nektar_wake.mat') / (1024*1024)
    print(f"Dataset ready. Size: {file_size:.2f} MB")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.autograd import grad
import time
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load cylinder wake data
print("\nLoading cylinder wake dataset (Re=100)...")
cylinder_data = loadmat('data/cylinder_nektar_wake.mat')

X_star = cylinder_data['X_star']
t_star = cylinder_data['t']
U_star = cylinder_data['U_star']
P_star = cylinder_data['p_star']

N = X_star.shape[0]
T = t_star.shape[0]

if len(U_star.shape) == 3:
    U_star_u = U_star[:, 0, :]
    U_star_v = U_star[:, 1, :]
else:
    U_star_u = U_star[:, 0:T]
    U_star_v = U_star[:, T:2*T]

print(f"Data loaded - Spatial points: {N}, Timesteps: {T}")
print(f"Time range: [{t_star.min():.3f}, {t_star.max():.3f}]")
print(f"Spatial extent: x=[{X_star[:,0].min():.2f}, {X_star[:,0].max():.2f}], "
      f"y=[{X_star[:,1].min():.2f}, {X_star[:,1].max():.2f}]")
print(f"Data shapes: u={U_star_u.shape}, v={U_star_v.shape}, p={P_star.shape}")

# PINN model
class ReynoldsAdaptivePINN(nn.Module):
    def __init__(self, hidden_size=100, num_layers=8, Re_input=False):
        super(ReynoldsAdaptivePINN, self).__init__()
        
        self.Re_input = Re_input
        input_dim = 4 if Re_input else 3
        
        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_layers - 2)
        ])
        self.fc_out = nn.Linear(hidden_size, 3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, y, t, Re=None):
        if self.Re_input and Re is not None:
            inputs = torch.cat([x, y, t, Re], dim=1)
        else:
            inputs = torch.cat([x, y, t], dim=1)
        
        h = torch.sin(self.fc_in(inputs))
        
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        
        out = self.fc_out(h)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        return u, v, p

# Physics-informed loss
class NavierStokesLoss(nn.Module):
    def __init__(self, Re=100, rho=1.0):
        super(NavierStokesLoss, self).__init__()
        self.Re = Re
        self.rho = rho
        self.nu = 1.0 / Re
    
    def compute_pde_residual(self, model, x, y, t, Re=None):
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p = model(x, y, t, Re)
        
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        f = u_t + u*u_x + v*u_y + (1/self.rho)*p_x - self.nu*(u_xx + u_yy)
        g = v_t + u*v_x + v*v_y + (1/self.rho)*p_y - self.nu*(v_xx + v_yy)
        h = u_x + v_y
        
        return f, g, h
    
    def forward(self, model, x_data, y_data, t_data, u_data, v_data, p_data,
                x_pde, y_pde, t_pde, Re_data=None, Re_pde=None):
        u_pred, v_pred, p_pred = model(x_data, y_data, t_data, Re_data)
        data_loss = torch.mean((u_pred - u_data)**2 + 
                               (v_pred - v_data)**2 + 
                               (p_pred - p_data)**2)
        
        f, g, h = self.compute_pde_residual(model, x_pde, y_pde, t_pde, Re_pde)
        pde_loss = torch.mean(f**2 + g**2 + h**2)
        
        return data_loss, pde_loss

# Data preparation
def prepare_training_data(X_star, t_star, U_star_u, U_star_v, P_star, 
                         n_samples=5000, seed=42):
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
    
    print(f"Training data: {n_samples} points ({100*n_samples/N_total:.2f}% of total)")
    
    return x_train, y_train, t_train, u_train, v_train, p_train

# Training function
def train_pinn(model, loss_fn, x_train, y_train, t_train, u_train, v_train, p_train,
               epochs=5000, lr=1e-3, Re_train=100, print_every=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5
    )
    
    history = {'data_loss': [], 'pde_loss': [], 'total_loss': []}
    
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    t_train = t_train.to(device)
    u_train = u_train.to(device)
    v_train = v_train.to(device)
    p_train = p_train.to(device)
    
    Re_tensor = torch.full_like(x_train, Re_train) if model.Re_input else None
    
    print(f"\nTraining PINN for Re={Re_train}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        data_loss, pde_loss = loss_fn(
            model, x_train, y_train, t_train, u_train, v_train, p_train,
            x_train, y_train, t_train, Re_tensor, Re_tensor
        )
        
        total_loss = data_loss + pde_loss
        total_loss.backward()
        optimizer.step()
        
        current_loss = total_loss.item()
        scheduler.step(current_loss)
        
        if current_loss < best_loss * 0.9:
            best_loss = current_loss
            print(f"  Learning rate adjusted: {optimizer.param_groups[0]['lr']:.2e}")
        
        history['data_loss'].append(data_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['total_loss'].append(current_loss)
        
        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:5d}/{epochs} | "
                  f"Data: {data_loss.item():.6f} | "
                  f"PDE: {pde_loss.item():.6f} | "
                  f"Total: {current_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    print(f"Training complete. Total time: {time.time()-start_time:.1f}s")
    print(f"Final loss: {current_loss:.6f}\n")
    
    return model, history

# Evaluation function
def evaluate_model(model, X_test, t_test, U_true, V_true, P_true, Re_test=100):
    model.eval()
    
    with torch.no_grad():
        x_test = torch.tensor(X_test[:, 0:1], dtype=torch.float32).to(device)
        y_test = torch.tensor(X_test[:, 1:2], dtype=torch.float32).to(device)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32).to(device)
        
        x_all = x_test.repeat(t_test_tensor.shape[0], 1)
        y_all = y_test.repeat(t_test_tensor.shape[0], 1)
        t_all = t_test_tensor.repeat_interleave(x_test.shape[0]).unsqueeze(1)
        
        Re_all = torch.full_like(x_all, Re_test) if model.Re_input else None
        
        u_pred, v_pred, p_pred = model(x_all, y_all, t_all, Re_all)
        
        u_pred = u_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
        v_pred = v_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
        p_pred = p_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
    
    u_error = np.linalg.norm(U_true - u_pred) / np.linalg.norm(U_true)
    v_error = np.linalg.norm(V_true - v_pred) / np.linalg.norm(V_true)
    p_error = np.linalg.norm(P_true - p_pred) / np.linalg.norm(P_true)
    
    print(f"\nEvaluation at Re={Re_test}")
    print(f"Relative L2 Error:")
    print(f"  u-velocity: {u_error:.6f} ({u_error*100:.2f}%)")
    print(f"  v-velocity: {v_error:.6f} ({v_error*100:.2f}%)")
    print(f"  pressure:   {p_error:.6f} ({p_error*100:.2f}%)\n")
    
    return u_pred, v_pred, p_pred, (u_error, v_error, p_error)

# Visualization
def plot_training_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    ax[0].semilogy(history['data_loss'], label='Data Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Data Loss')
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].semilogy(history['pde_loss'], label='PDE Loss', color='orange')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('PDE Loss')
    ax[1].grid(True)
    ax[1].legend()
    
    ax[2].semilogy(history['total_loss'], label='Total Loss', color='green')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Loss')
    ax[2].set_title('Total Loss')
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/pinn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history saved to Google Drive")

# Physics-based Reynolds scaling
def generate_reynolds_scaled_data(X, t, U_u, U_v, P, Re_base, Re_target):
    Re_ratio = Re_target / Re_base
    
    Cd_base = 1.2
    Cd_target = 1.2 / (1 + 0.3 * np.log10(Re_ratio))
    
    strouhal_factor = 1.0 + 0.05 * np.log10(Re_ratio)
    
    velocity_scale = np.sqrt(Re_ratio) * (Cd_target / Cd_base)
    
    U_scaled_u = U_u * velocity_scale
    U_scaled_v = U_v * velocity_scale * strouhal_factor
    P_scaled = P * velocity_scale**2
    
    print(f"\nGenerated Re={Re_target} from Re={Re_base}")
    print(f"  Velocity scale: {velocity_scale:.3f}")
    print(f"  Pressure scale: {velocity_scale**2:.3f}")
    print(f"  Cd: {Cd_target:.3f}")
    
    return U_scaled_u, U_scaled_v, P_scaled

# Execute baseline training
print("\nStarting PINN training for Reynolds extrapolation project")

n_train = int(N * T * 0.005)
x_tr, y_tr, t_tr, u_tr, v_tr, p_tr = prepare_training_data(
    X_star, t_star, U_star_u, U_star_v, P_star, n_samples=n_train
)

model = ReynoldsAdaptivePINN(hidden_size=128, num_layers=8, Re_input=False)
loss_fn = NavierStokesLoss(Re=100)

trained_model, training_history = train_pinn(
    model, loss_fn, x_tr, y_tr, t_tr, u_tr, v_tr, p_tr,
    epochs=3000, lr=1e-3, Re_train=100, print_every=300
)

u_pred, v_pred, p_pred, errors = evaluate_model(
    trained_model, X_star, t_star, U_star_u, U_star_v, P_star, Re_test=100
)

plot_training_history(training_history)

save_path = '/content/drive/MyDrive/pinn_re100_baseline.pt'
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'history': training_history,
    'errors': errors,
    'config': {
        'Re': 100,
        'hidden_size': 128,
        'num_layers': 8,
        'n_train': n_train
    }
}, save_path)

print(f"\nModel saved to: {save_path}")

# Reynolds extrapolation testing
print("\nReynolds number extrapolation analysis")

checkpoint = torch.load('/content/drive/MyDrive/pinn_re100_baseline.pt', 
                       weights_only=False,
                       map_location=device)

model = ReynoldsAdaptivePINN(hidden_size=128, num_layers=8, Re_input=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Loaded trained model from Re={checkpoint['config']['Re']}")
print(f"Training error: u={checkpoint['errors'][0]:.4f}, v={checkpoint['errors'][1]:.4f}, p={checkpoint['errors'][2]:.4f}")

def test_reynolds_extrapolation(model, X_test, t_test, U_true_u, U_true_v, P_true, 
                                Re_test_list, device):
    results = {
        'Reynolds': [],
        'u_error': [],
        'v_error': [],
        'p_error': [],
        'total_error': []
    }
    
    model.eval()
    
    with torch.no_grad():
        x_test = torch.tensor(X_test[:, 0:1], dtype=torch.float32).to(device)
        y_test = torch.tensor(X_test[:, 1:2], dtype=torch.float32).to(device)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32).to(device)
        
        x_all = x_test.repeat(t_test_tensor.shape[0], 1)
        y_all = y_test.repeat(t_test_tensor.shape[0], 1)
        t_all = t_test_tensor.repeat_interleave(x_test.shape[0]).unsqueeze(1)
        
        for Re_test in Re_test_list:
            print(f"\nTesting at Re={Re_test}...")
            
            Re_all = torch.full_like(x_all, Re_test) if model.Re_input else None
            
            u_pred, v_pred, p_pred = model(x_all, y_all, t_all, Re_all)
            
            u_pred = u_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
            v_pred = v_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
            p_pred = p_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
            
            u_error = np.linalg.norm(U_true_u - u_pred) / np.linalg.norm(U_true_u)
            v_error = np.linalg.norm(U_true_v - v_pred) / np.linalg.norm(U_true_v)
            p_error = np.linalg.norm(P_true - p_pred) / np.linalg.norm(P_true)
            total_error = (u_error + v_error + p_error) / 3
            
            results['Reynolds'].append(Re_test)
            results['u_error'].append(u_error)
            results['v_error'].append(v_error)
            results['p_error'].append(p_error)
            results['total_error'].append(total_error)
            
            print(f"  Relative L2 Errors:")
            print(f"    u-velocity: {u_error:.6f} ({u_error*100:.2f}%)")
            print(f"    v-velocity: {v_error:.6f} ({v_error*100:.2f}%)")
            print(f"    pressure:   {p_error:.6f} ({p_error*100:.2f}%)")
            print(f"    Average:    {total_error:.6f} ({total_error*100:.2f}%)")
    
    return results

print("\nTesting extrapolation performance")

Re_test_values = [100, 200, 400, 700, 1000, 1500]

print("\nNote: Currently testing with Re=100 data as baseline.")
print("For true extrapolation, you need CFD data at different Re values.\n")

results = test_reynolds_extrapolation(
    model, X_star, t_star, U_star_u, U_star_v, P_star,
    Re_test_values, device
)

# Visualize extrapolation results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(results['Reynolds'], np.array(results['u_error'])*100, 'o-', label='u-velocity', linewidth=2, markersize=8)
ax.plot(results['Reynolds'], np.array(results['v_error'])*100, 's-', label='v-velocity', linewidth=2, markersize=8)
ax.plot(results['Reynolds'], np.array(results['p_error'])*100, '^-', label='pressure', linewidth=2, markersize=8)
ax.plot(results['Reynolds'], np.array(results['total_error'])*100, 'd-', label='Average', 
        linewidth=3, color='black', markersize=10)
ax.set_xlabel('Reynolds Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative L2 Error (%)', fontsize=12, fontweight='bold')
ax.set_title('Extrapolation Error vs Reynolds Number', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Training Re')

ax = axes[0, 1]
ax.semilogy(results['Reynolds'], results['u_error'], 'o-', label='u-velocity', linewidth=2, markersize=8)
ax.semilogy(results['Reynolds'], results['v_error'], 's-', label='v-velocity', linewidth=2, markersize=8)
ax.semilogy(results['Reynolds'], results['p_error'], '^-', label='pressure', linewidth=2, markersize=8)
ax.set_xlabel('Reynolds Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative L2 Error (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Error Growth (Log Scale)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='red', linestyle='--', linewidth=2, alpha=0.5)

ax = axes[1, 0]
x_pos = np.arange(len(results['Reynolds']))
width = 0.25
ax.bar(x_pos - width, np.array(results['u_error'])*100, width, label='u-velocity', alpha=0.8)
ax.bar(x_pos, np.array(results['v_error'])*100, width, label='v-velocity', alpha=0.8)
ax.bar(x_pos + width, np.array(results['p_error'])*100, width, label='pressure', alpha=0.8)
ax.set_xlabel('Reynolds Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative L2 Error (%)', fontsize=12, fontweight='bold')
ax.set_title('Error Breakdown by Field', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results['Reynolds'])
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
ax.fill_between([0, 100], 0, 100, alpha=0.2, color='green', label='Training Range')
ax.fill_between([100, 700], 0, 100, alpha=0.2, color='yellow', label='Interpolation')
ax.fill_between([700, 1500], 0, 100, alpha=0.2, color='red', label='Extrapolation')
ax.plot(results['Reynolds'], np.array(results['total_error'])*100, 'ko-', 
        linewidth=3, markersize=10, label='Average Error')
ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% Threshold')
ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='20% Threshold')
ax.set_xlabel('Reynolds Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Error (%)', fontsize=12, fontweight='bold')
ax.set_title('Generalization Quality Zones', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1600)
ax.set_ylim(0, 30)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/reynolds_extrapolation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nExtrapolation analysis saved to Google Drive")

print("\nExtrapolation performance summary")
print(f"{'Re':>6} | {'u-error':>8} | {'v-error':>8} | {'p-error':>8} | {'avg-error':>8}")
print("-" * 60)
for i in range(len(results['Reynolds'])):
    print(f"{results['Reynolds'][i]:>6} | {results['u_error'][i]:>7.2%} | "
          f"{results['v_error'][i]:>7.2%} | {results['p_error'][i]:>7.2%} | "
          f"{results['total_error'][i]:>7.2%}")

breakdown_Re = None
for i, Re in enumerate(results['Reynolds']):
    if results['total_error'][i] > 0.20:
        breakdown_Re = Re
        break

if breakdown_Re:
    print(f"\nBreakdown Point: Re ~ {breakdown_Re}")
    print(f"(Average error exceeds 20%)")
else:
    print(f"\nNo breakdown detected up to Re = {max(results['Reynolds'])}")

best_field_idx = np.argmin([np.mean(results['u_error']), 
                            np.mean(results['v_error']), 
                            np.mean(results['p_error'])])
worst_field_idx = np.argmax([np.mean(results['u_error']), 
                             np.mean(results['v_error']), 
                             np.mean(results['p_error'])])
fields = ['u-velocity', 'v-velocity', 'pressure']

print(f"\nMost accurate field: {fields[best_field_idx]}")
print(f"Least accurate field: {fields[worst_field_idx]}")

df = pd.DataFrame(results)
df.to_csv('/content/drive/MyDrive/reynolds_extrapolation_results.csv', index=False)
print("\nResults saved to: reynolds_extrapolation_results.csv")

# Multi-Reynolds training
print("\nMulti-Reynolds PINN training")
print("Physics-based Reynolds scaling approach")

print("\nGenerating multi-Reynolds dataset")

reynolds_training = [200, 400, 600]
reynolds_extrapolation = [1000, 1500]

reynolds_data = {100: (U_star_u, U_star_v, P_star)}

for Re in reynolds_training + reynolds_extrapolation:
    U_u, U_v, P = generate_reynolds_scaled_data(
        X_star, t_star, U_star_u, U_star_v, P_star,
        Re_base=100, Re_target=Re
    )
    reynolds_data[Re] = (U_u, U_v, P)

print(f"\nDataset ready for Re = {sorted(reynolds_data.keys())}")

print("\nMulti-Reynolds PINN training")

trained_models = {}
all_results = []

for Re in reynolds_training:
    print(f"\nTraining: Re={Re}")
    
    U_u, U_v, P = reynolds_data[Re]
    
    n_train = int(X_star.shape[0] * t_star.shape[0] * 0.005)
    x_tr, y_tr, t_tr, u_tr, v_tr, p_tr = prepare_training_data(
        X_star, t_star, U_u, U_v, P, n_samples=n_train
    )
    
    model = ReynoldsAdaptivePINN(hidden_size=128, num_layers=8, Re_input=True)
    loss_fn = NavierStokesLoss(Re=Re)
    
    trained_model, history = train_pinn(
        model, loss_fn, x_tr, y_tr, t_tr, u_tr, v_tr, p_tr,
        epochs=1500, lr=1e-3, Re_train=Re, print_every=300
    )
    
    u_pred, v_pred, p_pred, errors = evaluate_model(
        trained_model, X_star, t_star, U_u, U_v, P, Re_test=Re
    )
    
    trained_models[Re] = {
        'model': trained_model,
        'history': history,
        'errors': errors
    }
    
    all_results.append({
        'Re': Re,
        'type': 'training',
        'u_error': errors[0],
        'v_error': errors[1],
        'p_error': errors[2]
    })
    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'errors': errors,
        'config': {'Re': Re, 'hidden_size': 128, 'num_layers': 8, 'Re_input': True}
    }, f'/content/drive/MyDrive/pinn_re{Re}_scaled.pt')

print(f"\nTraining complete for {len(trained_models)} models")

print("\nTesting Reynolds extrapolation")

for Re_test in reynolds_extrapolation:
    print(f"\nExtrapolation test: Re={Re_test}")
    
    U_u_test, U_v_test, P_test = reynolds_data[Re_test]
    
    closest_Re = min(trained_models.keys(), key=lambda x: abs(x - Re_test))
    print(f"Using model trained at Re={closest_Re}")
    
    test_model = trained_models[closest_Re]['model']
    test_model.eval()
    
    with torch.no_grad():
        x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
        y_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
        t_test_tensor = torch.tensor(t_star, dtype=torch.float32).to(device)
        
        x_all = x_test.repeat(t_test_tensor.shape[0], 1)
        y_all = y_test.repeat(t_test_tensor.shape[0], 1)
        t_all = t_test_tensor.repeat_interleave(x_test.shape[0]).unsqueeze(1)
        Re_all = torch.full_like(x_all, float(Re_test))
        
        u_pred, v_pred, p_pred = test_model(x_all, y_all, t_all, Re_all)
        
        u_pred = u_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
        v_pred = v_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
        p_pred = p_pred.reshape(t_test_tensor.shape[0], x_test.shape[0]).T.cpu().numpy()
    
    u_error = np.linalg.norm(U_u_test - u_pred) / np.linalg.norm(U_u_test)
    v_error = np.linalg.norm(U_v_test - v_pred) / np.linalg.norm(U_v_test)
    p_error = np.linalg.norm(P_test - p_pred) / np.linalg.norm(P_test)
    
    print(f"\nExtrapolation results:")
    print(f"  u-velocity: {u_error:.6f} ({u_error*100:.2f}%)")
    print(f"  v-velocity: {v_error:.6f} ({v_error*100:.2f}%)")
    print(f"  pressure:   {p_error:.6f} ({p_error*100:.2f}%)")
    
    all_results.append({
        'Re': Re_test,
        'type': 'extrapolation',
        'u_error': u_error,
        'v_error': v_error,
        'p_error': p_error,
        'trained_on': closest_Re
    })

print("\nGenerating results visualization")

df_results = pd.DataFrame(all_results)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :2])
training_df = df_results[df_results['type'] == 'training']
ax1.plot(training_df['Re'], training_df['u_error']*100, 'o-', label='u-velocity', linewidth=2, markersize=10)
ax1.plot(training_df['Re'], training_df['v_error']*100, 's-', label='v-velocity', linewidth=2, markersize=10)
ax1.plot(training_df['Re'], training_df['p_error']*100, '^-', label='pressure', linewidth=2, markersize=10)
ax1.set_xlabel('Reynolds Number', fontweight='bold', fontsize=12)
ax1.set_ylabel('Relative L2 Error (%)', fontweight='bold', fontsize=12)
ax1.set_title('Training Performance Across Reynolds Numbers', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
extrap_df = df_results[df_results['type'] == 'extrapolation']
x_pos = range(len(extrap_df))
width = 0.25
ax2.bar([p-width for p in x_pos], extrap_df['u_error']*100, width, label='u-vel', alpha=0.8)
ax2.bar(x_pos, extrap_df['v_error']*100, width, label='v-vel', alpha=0.8)
ax2.bar([p+width for p in x_pos], extrap_df['p_error']*100, width, label='pressure', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"Re={r}" for r in extrap_df['Re']])
ax2.set_ylabel('Error (%)', fontweight='bold')
ax2.set_title('Extrapolation Error', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[1, 0])
for Re in reynolds_training:
    ax3.semilogy(trained_models[Re]['history']['total_loss'], label=f'Re={Re}', linewidth=2)
ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('Total Loss (log)', fontweight='bold')
ax3.set_title('Training Convergence', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
all_re = sorted(df_results['Re'].unique())
for field, marker in [('u_error', 'o'), ('v_error', 's'), ('p_error', '^')]:
    errors = [df_results[df_results['Re']==r][field].values[0]*100 for r in all_re]
    ax4.plot(all_re, errors, marker=marker, label=field.replace('_error',''), linewidth=2, markersize=8)
ax4.axvline(x=max(reynolds_training), color='red', linestyle='--', linewidth=2, alpha=0.5, label='Training limit')
ax4.set_xlabel('Reynolds Number', fontweight='bold')
ax4.set_ylabel('Error (%)', fontweight='bold')
ax4.set_title('Error vs Reynolds Number', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
ax5.fill_between([0, 100], 0, 50, alpha=0.2, color='green', label='Base')
ax5.fill_between([100, 600], 0, 50, alpha=0.2, color='yellow', label='Training')
ax5.fill_between([600, 1500], 0, 50, alpha=0.2, color='red', label='Extrapolation')
avg_errors = [(df_results[df_results['Re']==r]['u_error'].values[0] +
               df_results[df_results['Re']==r]['v_error'].values[0] +
               df_results[df_results['Re']==r]['p_error'].values[0])/3*100
              for r in all_re]
ax5.plot(all_re, avg_errors, 'ko-', linewidth=3, markersize=10, label='Avg Error')
ax5.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
ax5.set_xlabel('Reynolds Number', fontweight='bold')
ax5.set_ylabel('Average Error (%)', fontweight='bold')
ax5.set_title('Generalization Quality', fontweight='bold')
ax5.set_xlim(0, 1600)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

table_data = []
for _, row in df_results.iterrows():
    table_data.append([
        f"{row['Re']:.0f}",
        row['type'].title(),
        f"{row['u_error']*100:.2f}%",
        f"{row['v_error']*100:.2f}%",
        f"{row['p_error']*100:.2f}%"
    ])

table = ax6.table(cellText=table_data,
                 colLabels=['Re', 'Type', 'u-error', 'v-error', 'p-error'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(df_results)):
    if df_results.iloc[i]['type'] == 'extrapolation':
        for j in range(5):
            table[(i+1, j)].set_facecolor('#ffcccc')

plt.suptitle('Multi-Reynolds PINN Training & Extrapolation Results',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/content/drive/MyDrive/multi_reynolds_complete_results.png', dpi=150, bbox_inches='tight')
plt.show()

df_results.to_csv('/content/drive/MyDrive/multi_reynolds_results.csv', index=False)

print("\nMulti-Reynolds training complete")

print(f"\nResults summary:")
print(f"  Training Re: {reynolds_training}")
print(f"  Extrapolation Re: {reynolds_extrapolation}")
print(f"\nAverage training error: {training_df['u_error'].mean()*100:.2f}%")
print(f"Average extrapolation error: {extrap_df['u_error'].mean()*100:.2f}%")

print(f"\nAll models saved to Google Drive")
print(f"Results saved to: multi_reynolds_results.csv")
print(f"Plots saved to: multi_reynolds_complete_results.png")
