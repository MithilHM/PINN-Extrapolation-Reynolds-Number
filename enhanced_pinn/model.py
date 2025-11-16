import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F


# ------------------------------
# Utility: compute dy/dx
# ------------------------------
def calc_grad(y, x) -> Tensor:
    return autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


# ------------------------------
# Transformer-style FFN block
# ------------------------------
class FfnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inter = 4 * dim
        self.fc1 = nn.Linear(dim, inter)
        self.fc2 = nn.Linear(inter, dim)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(self.fc2(h))
        return x + h


# ------------------------------
# PINN Model
# ------------------------------
class Pinn(nn.Module):
    def __init__(self, min_x, max_x):
        super().__init__()

        # store as torch tensors (model parameters or buffers)
        self.register_buffer("MIN_X", torch.tensor(min_x, dtype=torch.float32))
        self.register_buffer("MAX_X", torch.tensor(max_x, dtype=torch.float32))

        # Encoder
        self.hidden = 128
        self.blocks = 8
        self.first = nn.Linear(3, self.hidden)
        self.last = nn.Linear(self.hidden, 2)

        self.ffn_blocks = nn.ModuleList([FfnBlock(self.hidden) for _ in range(self.blocks)])

        # Trainable NS parameters
        self.lambda1 = nn.Parameter(torch.tensor(1.0))   # inertia
        self.lambda2 = nn.Parameter(torch.tensor(0.01))  # viscosity

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------
    # Forward + PDE residual computation
    # ------------------------------
    def forward(self, x, y, t, p=None, u=None, v=None):
        inp = torch.stack([x, y, t], dim=1)
        inp = 2.0 * (inp - self.MIN_X) / (self.MAX_X - self.MIN_X) - 1.0

        h = self.first(inp)
        for blk in self.ffn_blocks:
            h = blk(h)
        h = self.last(h)

        psi = h[:, 0]
        p_pred = h[:, 1]

        # velocity from streamfunction
        u_pred = calc_grad(psi, y)
        v_pred = -calc_grad(psi, x)

        # derivatives
        u_t = calc_grad(u_pred, t)
        u_x = calc_grad(u_pred, x)
        u_y = calc_grad(u_pred, y)
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        v_t = calc_grad(v_pred, t)
        v_x = calc_grad(v_pred, x)
        v_y = calc_grad(v_pred, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        p_x = calc_grad(p_pred, x)
        p_y = calc_grad(p_pred, y)

        # ------------------------------
        # Correct TGV Navier-Stokes residuals (NO GRAVITY)
        # ------------------------------
        f_u = (
            self.lambda1 * (u_t + u_pred * u_x + v_pred * u_y)
            + p_x
            - self.lambda2 * (u_xx + u_yy)
        )

        f_v = (
            self.lambda1 * (v_t + u_pred * v_x + v_pred * v_y)
            + p_y
            - self.lambda2 * (v_xx + v_yy)
        )

        loss, losses = self.loss_fn(u, v, u_pred, v_pred, f_u, f_v)

        return {
            "preds": torch.stack([p_pred, u_pred, v_pred], dim=1),
            "loss": loss,
            "losses": losses,
            "f_u": f_u.detach(),
            "f_v": f_v.detach(),
        }

    # ------------------------------
    # Improved GBLW + Soft PDE scaling
    # ------------------------------
    def loss_fn(self, u, v, u_pred, v_pred, f_u_pred, f_v_pred):

        # supervised losses
        u_loss = F.mse_loss(u_pred, u)
        v_loss = F.mse_loss(v_pred, v)

        # PDE losses
        f_u_loss = F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred))
        f_v_loss = F.mse_loss(f_v_pred, torch.zeros_like(f_v_pred))

        # -------- Soft PDE scaling (critical!) --------
        pde_scale = 0.1
        f_u_loss = pde_scale * f_u_loss
        f_v_loss = pde_scale * f_v_loss

        # -------- Lightweight GBLW (stable & fast) --------
        eps = 1e-8
        params = [p for p in self.parameters() if p.requires_grad]

        def gnorm(loss):
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return sum((gi.norm() for gi in g if gi is not None)) + eps

        gn_u = gnorm(u_loss)
        gn_v = gnorm(v_loss)
        gn_fu = gnorm(f_u_loss)
        gn_fv = gnorm(f_v_loss)

        w_u = 1 / gn_u
        w_v = 1 / gn_v
        w_fu = 1 / gn_fu
        w_fv = 1 / gn_fv

        s = w_u + w_v + w_fu + w_fv
        w_u /= s; w_v /= s; w_fu /= s; w_fv /= s

        # Final loss
        loss = (
            w_u * u_loss +
            w_v * v_loss +
            w_fu * f_u_loss +
            w_fv * f_v_loss
        )

        return loss, {
            "u_loss": u_loss.detach(),
            "v_loss": v_loss.detach(),
            "f_u_loss": f_u_loss.detach(),
            "f_v_loss": f_v_loss.detach(),
            "w_u": w_u.detach(),
            "w_v": w_v.detach(),
            "w_fu": w_fu.detach(),
            "w_fv": w_fv.detach(),
        }
