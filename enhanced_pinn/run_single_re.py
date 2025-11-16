from pathlib import Path
from model import Pinn
from trainer import Trainer
from data import get_dataset


def run(Re, filename, epochs=10, batch_size=32, samples_per_ep=1000):

    print("\n===============================")
    print(f"   Training Re = {Re}")
    print(f"   Epochs = {epochs}")
    print("===============================\n")

    # load dataset
    train_data, test_data, min_x, max_x = get_dataset(Path(f"data/{filename}"))

    # output directory
    out_dir = Path(f"result/lowRe{Re}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # build model
    model = Pinn(min_x, max_x)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    # ---- MATCHING Trainer EXACT constructor ----
    trainer = Trainer(
        model,          # 1st positional
        train_data,     # 2nd positional
        batch_size,     # 3rd positional
        0.20,           # sample_ratio
        0.02,           # topk_frac
        2,              # ras_interval
        "cuda"          # device
    )

    # ---- call trainer.train ----
    trainer.train(
        epochs=epochs,
        samples_per_epoch=samples_per_ep,
        out_dir=out_dir
    )

    print("\nTraining complete!")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    run(400, "Re400.jsonl")
