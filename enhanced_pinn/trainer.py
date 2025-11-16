"""
Patched trainer.py for PINN project.

Features:
- GPU/CPU auto device selection
- Fast Residual-Adaptive Sampling (RAS) with:
    * subsampling of candidate pool
    * small top_k fraction
    * topk on CPU (detached)
    * configurable ras_interval
- Saves checkpoints as pure state_dict (ckpt.pt) with lambda1/lambda2/loss
- Simple train loop compatible with run_single_re.run(...)
"""

import os
import json
import math
import time
import random
import shutil
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

# import your PINN model
from model import Pinn

# -----------------------
# Simple Dataset wrapper
# -----------------------
class JsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []
        with open(jsonl_path, "r") as f:
            header = json.loads(f.readline())  # ["t","x","y","p","u","v"]
            for line in f:
                row = json.loads(line)
                # store as numpy floats for efficiency
                self.rows.append((
                    float(row[0]),  # t
                    float(row[1]),  # x
                    float(row[2]),  # y
                    float(row[3]),  # p
                    float(row[4]),  # u
                    float(row[5])   # v
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        t,x,y,p,u,v = self.rows[idx]
        # return raw floats; conversion to torch happens in collate
        return {"t": t, "x": x, "y": y, "p": p, "u": u, "v": v}

# Collate to tensors (non-batched shapes)
def collate_fn(batch):
    b = batch
    t = torch.tensor([x["t"] for x in b], dtype=torch.float32)
    x = torch.tensor([x["x"] for x in b], dtype=torch.float32)
    y = torch.tensor([x["y"] for x in b], dtype=torch.float32)
    p = torch.tensor([x["p"] for x in b], dtype=torch.float32)
    u = torch.tensor([x["u"] for x in b], dtype=torch.float32)
    v = torch.tensor([x["v"] for x in b], dtype=torch.float32)
    return {"t": t, "x": x, "y": y, "p": p, "u": u, "v": v}

# -----------------------
# Trainer class
# -----------------------
class Trainer:
    def __init__(
        self,
        jsonl_path,
        result_dir="result/lowRe",
        device=None,
        ras_sample_ratio=0.2,   # subsample fraction for RAS candidate pool
        ras_topk_frac=0.02,     # fraction of subsample to pick (2% default)
        ras_interval=2,         # run RAS every N epochs
        lr=1e-3,
    ):
        self.jsonl_path = jsonl_path
        self.dataset = JsonlDataset(jsonl_path)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print("Using device:", self.device)

        # RAS settings
        self.ras_sample_ratio = float(ras_sample_ratio)
        self.ras_topk_frac = float(ras_topk_frac)
        self.ras_interval = int(ras_interval)

        self.lr = lr

    def build_model(self):
        # compute min/max for normalization from dataset (t,x,y)
        arr = np.array([[r[0], r[1], r[2]] for r in self.dataset.rows], dtype=np.float32)
        min_x = arr.min(axis=0)
        max_x = arr.max(axis=0)
        model = Pinn(min_x, max_x)
        return model

    def save_ckpt(self, model_state_dict, epoch, loss, out_folder):
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        ckpt = {}
        # save state_dict (pure tensors)
        ckpt.update(model_state_dict)
        # if lambda params are present in model_state_dict they are saved already (names like 'lambda1' may or may not exist)
        # also save scalar loss for convenience
        ckpt["loss"] = float(loss)
        torch.save(ckpt, out_folder / "ckpt.pt")
        # also write loss_history.json (simple append)
        with open(out_folder / "loss_history.json", "w") as f:
            json.dump({"epoch": epoch, "loss": float(loss)}, f)

    def compute_batch_loss(self, model, batch):
        # model expects tensors with requires_grad True for x,y,t
        x = batch["x"].to(self.device).requires_grad_(True)
        y = batch["y"].to(self.device).requires_grad_(True)
        t = batch["t"].to(self.device).requires_grad_(True)
        p = batch["p"].to(self.device)
        u = batch["u"].to(self.device)
        v = batch["v"].to(self.device)

        # forward with GT
        out = model(x, y, t, p=p, u=u, v=v)

        # out is dict with 'loss' scalar (already computed inside model)
        loss = out["loss"]
        return loss

    # Fast RAS implementation (no gradient leaks)
    def run_ras(self, model, epoch, full_indices):
        """
        - model: on-device model
        - full_indices: numpy indices (0..N-1) of the dataset currently used
        Returns: new_indices (numpy array of indices)
        """
        N = len(full_indices)
        if N == 0:
            return full_indices

        # Step 1: Subsample candidate pool (random)
        sub_N = max(1, int(N * self.ras_sample_ratio))
        # random subset indices in the dataset index space
        sub_idx = np.random.choice(full_indices, size=sub_N, replace=False)

        # Build a batch of candidate tensors (as one big tensor) to compute residuals efficiently
        t_arr = np.array([self.dataset.rows[i][0] for i in sub_idx], dtype=np.float32)
        x_arr = np.array([self.dataset.rows[i][1] for i in sub_idx], dtype=np.float32)
        y_arr = np.array([self.dataset.rows[i][2] for i in sub_idx], dtype=np.float32)
        p_arr = np.array([self.dataset.rows[i][3] for i in sub_idx], dtype=np.float32)
        u_arr = np.array([self.dataset.rows[i][4] for i in sub_idx], dtype=np.float32)
        v_arr = np.array([self.dataset.rows[i][5] for i in sub_idx], dtype=np.float32)

        # convert to torch on device with requires_grad for PINN residual evaluation
        x_t = torch.tensor(x_arr, dtype=torch.float32, device=self.device, requires_grad=True)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=self.device, requires_grad=True)
        t_t = torch.tensor(t_arr, dtype=torch.float32, device=self.device, requires_grad=True)

        # compute residuals via model forward (the model computes f_u/f_v)
        model.eval()
        with torch.enable_grad():
            out = model(x_t, y_t, t_t, p=None, u=torch.zeros_like(x_t), v=torch.zeros_like(x_t))

        # out["f_u"], out["f_v"] are tensors; compute a scalar residual measure per point
        f_u = out["f_u"]  # shape [sub_N]
        f_v = out["f_v"]
        # safety: ensure they are floating tensors
        res = (f_u.detach()**2 + f_v.detach()**2).sum(dim=0) if f_u.dim() > 1 else (f_u.detach()**2 + f_v.detach()**2)
        # res is a tensor on device -> move to cpu but detached
        res_cpu = res.detach().cpu()

        # Step 2: select top-k on CPU
        top_k = max(1, int(self.ras_topk_frac * sub_N))
        # get indices within sub_idx corresponding to largest residuals
        topk_res = torch.topk(res_cpu, k=top_k)
        topk_indices_in_sub = topk_res.indices.numpy()  # safe: res_cpu is detached on CPU
        chosen_dataset_indices = sub_idx[topk_indices_in_sub]  # picks dataset indices

        # Build new index set: keep existing ones but replace some low-residual ones with chosen high-residual ones
        # Heuristic: keep a large fraction of previous indices, swap a small fraction
        keep_ratio = 0.8
        keep_n = max(1, int(keep_ratio * N))
        keep_idx = np.random.choice(full_indices, size=keep_n, replace=False)
        new_indices = np.unique(np.concatenate([keep_idx, chosen_dataset_indices])).astype(int)

        # ensure new_indices length not exceed N
        if len(new_indices) > N:
            new_indices = new_indices[:N]

        print(f"[RAS] epoch={epoch} sub_N={sub_N} top_k={top_k} -> new size={len(new_indices)}")
        return new_indices

    def train(
        self,
        re,
        epochs=10,
        batch_size=32,
        samples_per_ep=1000,
        result_subdir=None,
        save_every=1,
        verbose=True,
    ):
        """
        re: Reynolds or identifier (used for naming)
        epochs, batch_size: training hyperparams
        samples_per_ep: number of collocation/supervised samples per epoch
        """

        # create result folder for this run
        out_dir = self.result_dir / f"lowRe{re}"
        if result_subdir:
            out_dir = self.result_dir / result_subdir
        if out_dir.exists():
            # keep existing results, but user can delete if they want fresh run
            pass
        out_dir.mkdir(parents=True, exist_ok=True)

        # instantiate model and optimizer
        model = self.build_model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # initialize dataset indices (full pool)
        pool_N = len(self.dataset)
        pool_indices = np.arange(pool_N)

        # initial sample indices for epoch 0: random
        epoch_indices = np.random.choice(pool_indices, size=min(samples_per_ep, pool_N), replace=False)

        for epoch in range(epochs):
            t0 = time.time()
            model.train()

            # create DataLoader from epoch_indices (create in-memory batch arrays for speed)
            selected_rows = [self.dataset.rows[i] for i in epoch_indices]
            # build torch tensors for the epoch
            t_arr = torch.tensor([r[0] for r in selected_rows], dtype=torch.float32, device=self.device)
            x_arr = torch.tensor([r[1] for r in selected_rows], dtype=torch.float32, device=self.device)
            y_arr = torch.tensor([r[2] for r in selected_rows], dtype=torch.float32, device=self.device)
            p_arr = torch.tensor([r[3] for r in selected_rows], dtype=torch.float32, device=self.device)
            u_arr = torch.tensor([r[4] for r in selected_rows], dtype=torch.float32, device=self.device)
            v_arr = torch.tensor([r[5] for r in selected_rows], dtype=torch.float32, device=self.device)

            # iterate mini-batches
            num_samples = x_arr.shape[0]
            perm = torch.randperm(num_samples)
            epoch_loss = 0.0
            steps = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]

                batch = {
                    "t": t_arr[idx],
                    "x": x_arr[idx],
                    "y": y_arr[idx],
                    "p": p_arr[idx],
                    "u": u_arr[idx],
                    "v": v_arr[idx],
                }

                optimizer.zero_grad()
                loss = self.compute_batch_loss(model, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().numpy())
                steps += 1

            epoch_loss = epoch_loss / max(1, steps)
            dt = time.time() - t0

            if verbose:
                print(f"[Epoch {epoch:02d}] loss={epoch_loss:.6e} samples={len(epoch_indices)} time={dt:.2f}s")

            # Save checkpoint (state_dict only) every save_every epochs
            ckpt_folder = out_dir / f"ckpt-{epoch}"
            ckpt_folder.mkdir(parents=True, exist_ok=True)
            model_sd = model.state_dict()
            # we will save state_dict (flat) and optionally lambdas + loss inside the same dict for convenience
            save_dict = {}
            for k, v in model_sd.items():
                save_dict[k] = v.detach().cpu()
            # also save lambda1/lambda2 if present as parameters
            try:
                save_dict["lambda1"] = model.lambda1.detach().cpu()
                save_dict["lambda2"] = model.lambda2.detach().cpu()
            except Exception:
                pass
            save_dict["loss"] = float(epoch_loss)
            torch.save(save_dict, ckpt_folder / "ckpt.pt")
            # small metadata file
            with open(ckpt_folder / "metrics.json", "w") as f:
                json.dump({"loss": epoch_loss}, f)

            # RAS: only run on configured interval (and not on last epoch)
            if (epoch % self.ras_interval == 0) and (epoch < epochs - 1):
                # compute new epoch_indices using RAS
                epoch_indices = self.run_ras(model, epoch, pool_indices)
                # make sure we have exactly samples_per_ep indices (pad or trim)
                if len(epoch_indices) < samples_per_ep:
                    add_n = samples_per_ep - len(epoch_indices)
                    extra = np.random.choice(pool_indices, size=add_n, replace=False)
                    epoch_indices = np.concatenate([epoch_indices, extra])
                elif len(epoch_indices) > samples_per_ep:
                    epoch_indices = epoch_indices[:samples_per_ep]
            else:
                # random sample for next epoch
                epoch_indices = np.random.choice(pool_indices, size=min(samples_per_ep, pool_N), replace=False)

        print("Training finished. Results in:", out_dir)


# -----------------------
# Convenience run function (compatible with your previous API)
# -----------------------
def run(re, jsonl_path, epochs=10, batch_size=32, samples_per_ep=1000, result_dir="result", **kwargs):
    """
    Wrapper used by your previous command:
    py -c "from run_single_re import run; run(400,'Re400.jsonl',epochs=10, batch_size=32, samples_per_ep=1000)"
    """
    trainer = Trainer(
        jsonl_path=jsonl_path,
        result_dir=result_dir,
        ras_sample_ratio=kwargs.get("ras_sample_ratio", 0.2),
        ras_topk_frac=kwargs.get("ras_topk_frac", 0.02),
        ras_interval=kwargs.get("ras_interval", 2),
        lr=kwargs.get("lr", 1e-3),
    )
    trainer.train(re, epochs=epochs, batch_size=batch_size, samples_per_ep=samples_per_ep, result_subdir=f"lowRe{re}")

# If this file is executed directly for quick testing:
if __name__ == "__main__":
    # example quick test (small)
    run(400, "data/Re400.jsonl", epochs=1, batch_size=16, samples_per_ep=200, result_dir="result")
