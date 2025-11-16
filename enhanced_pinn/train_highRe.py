from pathlib import Path
import torch
import numpy as np
import random

from model import Pinn
from data import PinnDataset, get_dataset
from trainer import Trainer


def main():
    # ----------------------------
    # 1. Random seeds
    # ----------------------------
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----------------------------
    # 2. High-Re dataset
    # ----------------------------
    train_data, test_data, min_x, max_x = get_dataset(Path("data/Re1000.jsonl"))

    # ----------------------------
    # 3. Create model
    # ----------------------------
    model = Pinn(min_x, max_x)
    trainer = Trainer(
        model,
        output_dir=Path("result/highRe1000"),
        num_epochs=40,
        batch_size=32
    )
    trainer.samples_per_ep = 5000

    # ----------------------------
    # 4. Load LOW-Re trained weights
    # ----------------------------
    lowRe_ckpt_path = Path("result/lowRe200/ckpt-39")  # last epoch from low-Re
    trainer.load_ckpt(lowRe_ckpt_path)

    # ----------------------------
    # 5. Use smaller learning rate for fine-tuning
    # ----------------------------
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------
    # 6. Train on HIGH-Re (transfer learning)
    # ----------------------------
    trainer.train(train_data, do_resume=False)

    print("\n==== Transfer Learning Complete ====")
    print("High-Re model saved at:", trainer.get_last_ckpt_dir())


if __name__ == "__main__":
    main()
