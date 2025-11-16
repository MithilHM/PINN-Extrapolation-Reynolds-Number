# analyze_ckpt.py
# Usage example:
#   py analyze_ckpt.py --ckpt_ckpt result/lowRe200/ckpt-9/ckpt.pt --loss_json result/lowRe200/ckpt-9/loss_history.json

import argparse, json
from pathlib import Path
import numpy as np

def safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError(f"Matplotlib import error: {e}")

def safe_import_torch():
    try:
        import torch
        return torch
    except Exception as e:
        raise RuntimeError(f"Torch import error: {e}")

def load_loss_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Flatten nested lists if needed
    flat = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, (int, float)):
                flat.append(item)
    else:
        raise RuntimeError("loss_history.json has unexpected format")

    return np.array(flat, dtype=float)

def plot_loss(arr, out_path):
    plt = safe_import_matplotlib()
    import pandas as pd

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(arr, linewidth=1, label="loss (per step)")

    win = max(1, len(arr) // 20)
    smoothed = pd.Series(arr).rolling(win, min_periods=1).mean().values
    ax.plot(smoothed, linewidth=2, label=f"smooth (window={win})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_hist(arr, out_path):
    plt = safe_import_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(arr, bins=30)
    ax.set_title("Loss Histogram")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def extract_lambdas(model_state):
    lambda1 = None
    lambda2 = None

    for k, v in model_state.items():
        name = k.lower()
        try:
            if "lambda1" in name:
                lambda1 = float(v.cpu().item())
            if "lambda2" in name:
                lambda2 = float(v.cpu().item())
        except:
            pass

    return lambda1, lambda2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_ckpt", type=str, required=True, help="Path to ckpt.pt")
    parser.add_argument("--loss_json", type=str, required=True, help="Path to loss_history.json")
    parser.add_argument("--out_dir", type=str, default="result/ckpt_analysis", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load loss history
    print("Loading loss history...")
    arr = load_loss_json(args.loss_json)

    # Plot loss + histogram
    print("Creating plots...")
    plot_loss(arr, out_dir / "loss_history.png")
    plot_hist(arr, out_dir / "loss_hist.png")

    # Load model and extract lambda values
    print("Extracting lambda1/lambda2...")
    torch = safe_import_torch()
    state = torch.load(args.ckpt_ckpt, map_location="cpu")

    # If checkpoint is nested
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    lambda1, lambda2 = extract_lambdas(state)

    # Save summary
    summary = {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "loss_mean": float(np.mean(arr)),
        "loss_min": float(np.min(arr)),
        "loss_max": float(np.max(arr)),
        "loss_std": float(np.std(arr)),
        "loss_count": int(arr.shape[0])
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Analysis Complete ===")
    print("Files saved in:", out_dir)
    print("Upload loss_history.png, loss_hist.png, and summary.json back to ChatGPT.\n")

if __name__ == "__main__":
    main()
