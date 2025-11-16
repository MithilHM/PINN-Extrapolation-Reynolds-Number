import os
import json
import glob
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# your PINN model
from model import Pinn


# ---------------------------------------------------
# Load JSONL ["t","x","y","p","u","v"]
# ---------------------------------------------------
def load_jsonl(path):
    data = {"t": [], "x": [], "y": [], "p": [], "u": [], "v": []}
    with open(path, "r") as f:
        header = json.loads(f.readline())
        for line in f:
            row = json.loads(line)
            data["t"].append(row[0])
            data["x"].append(row[1])
            data["y"].append(row[2])
            data["p"].append(row[3])
            data["u"].append(row[4])
            data["v"].append(row[5])
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


# ---------------------------------------------------
# Plot helper
# ---------------------------------------------------
def plot_curve(values, path, label):
    plt.figure()
    plt.plot(values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------
# Evaluate one checkpoint  (FIXED)
# ---------------------------------------------------
def evaluate_ckpt(ckpt_file, test, min_x, max_x):

    sd = torch.load(ckpt_file, map_location="cpu")

    # unwrap state_dict format if needed
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # instantiate PINN
    model = Pinn(min_x, max_x)
    model.eval()

    # load weights
    try:
        model.load_state_dict(sd, strict=False)
    except:
        filtered = {k: v for k, v in sd.items() if k in model.state_dict()}
        model.load_state_dict(filtered, strict=False)

    # restore lambdas
    if "lambda1" in sd:
        model.lambda1.data = sd["lambda1"]
    if "lambda2" in sd:
        model.lambda2.data = sd["lambda2"]

    # tensors with grad
    x = torch.tensor(test["x"], dtype=torch.float32, requires_grad=True)
    y = torch.tensor(test["y"], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(test["t"], dtype=torch.float32, requires_grad=True)

    # ground truth (no grad)
    u = torch.tensor(test["u"], dtype=torch.float32)
    v = torch.tensor(test["v"], dtype=torch.float32)
    p = torch.tensor(test["p"], dtype=torch.float32)

    # IMPORTANT: no torch.no_grad(), PINN needs autograd
    out = model(x, y, t)

    # predictions
    preds = out["preds"]
    preds_np = preds.detach().cpu().numpy()

    p_pred = preds_np[:, 0]
    u_pred = preds_np[:, 1]
    v_pred = preds_np[:, 2]

    # detach true values
    p_np = p.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()

    # RMSE
    rmse_p = float(np.sqrt(np.mean((p.detach().cpu().numpy() - p_pred) ** 2)))
    rmse_u = float(np.sqrt(np.mean((u.detach().cpu().numpy() - u_pred) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v.detach().cpu().numpy() - v_pred) ** 2)))

    # PDE residuals
    fu = out["f_u"].detach().cpu().numpy()
    fv = out["f_v"].detach().cpu().numpy()

    mean_fu = float(np.mean(fu))
    mean_fv = float(np.mean(fv))

    # metadata
    lambda1 = float(sd["lambda1"]) if "lambda1" in sd else np.nan
    lambda2 = float(sd["lambda2"]) if "lambda2" in sd else np.nan
    loss_val = float(sd["loss"]) if "loss" in sd else np.nan

    return rmse_u, rmse_v, rmse_p, mean_fu, mean_fv, lambda1, lambda2, loss_val


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: py analyze_all_epochs.py <result_dir> <jsonl>")
        sys.exit(1)

    result_dir = sys.argv[1]
    test_path = sys.argv[2]

    print("Loading test set:", test_path)
    test = load_jsonl(test_path)

    # compute min/max for normalization
    arr3 = np.stack([test["t"], test["x"], test["y"]], axis=1)
    min_x = arr3.min(axis=0).astype(np.float32)
    max_x = arr3.max(axis=0).astype(np.float32)

    # checkpoints
    ckpts = sorted(glob.glob(os.path.join(result_dir, "ckpt-*")))
    print("Found", len(ckpts), "epochs.")

    out_dir = os.path.join(result_dir, "analysis_all_epochs")
    os.makedirs(out_dir, exist_ok=True)

    all_rmse_u = []
    all_rmse_v = []
    all_rmse_p = []
    all_fu = []
    all_fv = []
    lambda1_list = []
    lambda2_list = []
    loss_list = []

    for ckpt in ckpts:
        ckpt_file = os.path.join(ckpt, "ckpt.pt")
        if not os.path.exists(ckpt_file):
            print("Skipping:", ckpt_file)
            continue

        print("Evaluating:", ckpt_file)

        try:
            rmse_u, rmse_v, rmse_p, fu, fv, lam1, lam2, loss_val = evaluate_ckpt(
                ckpt_file, test, min_x, max_x)
        except Exception as e:
            print("ERROR:", e)
            continue

        all_rmse_u.append(rmse_u)
        all_rmse_v.append(rmse_v)
        all_rmse_p.append(rmse_p)
        all_fu.append(fu)
        all_fv.append(fv)
        lambda1_list.append(lam1)
        lambda2_list.append(lam2)
        loss_list.append(loss_val)

    # save CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,rmse_u,rmse_v,rmse_p,mean_fu,mean_fv,lambda1,lambda2,loss\n")
        for i in range(len(all_rmse_u)):
            f.write(
                f"{i},{all_rmse_u[i]},{all_rmse_v[i]},{all_rmse_p[i]},"
                f"{all_fu[i]},{all_fv[i]},{lambda1_list[i]},{lambda2_list[i]},{loss_list[i]}\n"
            )

    # plots
    plot_curve(all_rmse_u, os.path.join(out_dir, "rmse_u.png"), "RMSE u")
    plot_curve(all_rmse_v, os.path.join(out_dir, "rmse_v.png"), "RMSE v")
    plot_curve(all_rmse_p, os.path.join(out_dir, "rmse_p.png"), "RMSE p")
    plot_curve(all_fu, os.path.join(out_dir, "mean_fu.png"), "Mean f_u")
    plot_curve(all_fv, os.path.join(out_dir, "mean_fv.png"), "Mean f_v")
    plot_curve(lambda1_list, os.path.join(out_dir, "lambda1.png"), "lambda1")
    plot_curve(lambda2_list, os.path.join(out_dir, "lambda2.png"), "lambda2")
    plot_curve(loss_list, os.path.join(out_dir, "loss.png"), "Loss")

    print("Analysis complete!")
    print("Results saved to:", out_dir)


if __name__ == "__main__":
    main()
