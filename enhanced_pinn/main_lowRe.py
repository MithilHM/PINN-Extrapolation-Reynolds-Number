from pathlib import Path
import random
import numpy as np
import torch

from model import Pinn
from data import PinnDataset, get_dataset
from trainer import Trainer


def train_single_Re(re_value, filename):
    print(f"\n===============================")
    print(f" Training Low-Re = {re_value}")
    print(f"===============================\n")

    # Load dataset
    train_data, test_data, min_x, max_x = get_dataset(Path(f"data/{filename}"))

    # Build model
    model = Pinn(min_x, max_x)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params}")

    # Trainer config
    trainer = Trainer(
        model,
        output_dir=Path(f"result/lowRe{re_value}"),
        num_epochs=40,
        batch_size=32
    )
    trainer.samples_per_ep = 5000

    # Train
    trainer.train(train_data)

    print(f"Finished training Re = {re_value}")
    print(f"Saved checkpoint at: {trainer.get_last_ckpt_dir()}")


def main():
    torch.random.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Train low Re values one by one
    train_single_Re(200, "Re200.jsonl")
    train_single_Re(400, "Re400.jsonl")
    train_single_Re(600, "Re600.jsonl")


if __name__ == "__main__":
    main()
