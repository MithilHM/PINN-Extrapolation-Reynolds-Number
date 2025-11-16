from pathlib import Path
import torch
import numpy as np
import random

from model import Pinn
from data import PinnDataset, get_dataset   # if using JSONL
from trainer import Trainer


def main():
    # ----------------------------
    # 1. Set random seeds
    # ----------------------------
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----------------------------
    # 2. Load LOW-Re dataset
    # Replace with your actual dataset
    # ----------------------------
    # Example: data/Re200.jsonl
    train_data, test_data, min_x, max_x = get_dataset(Path("data/Re200.jsonl"))

    # ----------------------------
    # 3. Create model
    # ----------------------------
    model = Pinn(min_x, max_x)

    # ----------------------------
    # 4. Trainer (40 epochs recommended)
    # ----------------------------
    trainer = Trainer(
        model,
        output_dir=Path("result/lowRe200"),
        num_epochs=40,
        batch_size=32
    )
    trainer.samples_per_ep = 5000  # More samples for real training

    # ----------------------------
    # 5. Train low-Re model
    # ----------------------------
    trainer.train(train_data)

    print("\n==== Training Low Re Complete ====")
    print("Saved at:", trainer.get_last_ckpt_dir())


if __name__ == "__main__":
    main()
