# PINN Reynolds Number Extrapolation

This repository contains implementations and experiments for studying the extrapolation capabilities of Physics-Informed Neural Networks (PINNs) across different Reynolds numbers in fluid dynamics problems.

## Project Overview

This project investigates how PINNs generalize to unseen Reynolds numbers, particularly focusing on the transition from interpolation (Re ≤ 700) to extrapolation (Re > 700) regimes. The work compares PINN performance against standard neural networks for Navier-Stokes equation solving across a range of Reynolds numbers.

## Repository Structure

```
├── enhanced_pinn/           # Enhanced PINN implementation with advanced features
│   ├── data/               # Training datasets (Reynolds number specific)
│   │   ├── Re200.jsonl
│   │   ├── Re400.jsonl
│   │   └── Re600.jsonl
│   ├── result/             # Training results and checkpoints
│   ├── model.py            # PINN model architecture
│   ├── trainer.py          # Training loop implementation
│   ├── data.py             # Data loading and preprocessing
│   ├── train_lowRe.py      # Training script for low Reynolds numbers
│   ├── train_highRe.py     # Training script for high Reynolds numbers
│   └── ...
├── standardpinn.py          # Standard PINN implementation
└── standard_nn.py           # Standard neural network baseline
```

## Key Features

### Enhanced PINN Implementation
- Transformer-style FFN blocks for improved performance
- Residual Adaptive Sampling (RAS) for efficient training
- Gradient-based loss weighting (GBLW) for balanced physics constraints
- Soft PDE scaling for stable training

### Data
The project uses the cylinder wake dataset from the original PINNs paper, containing:
- Spatial coordinates (x, y)
- Time (t)
- Velocity components (u, v)
- Pressure (p)

Datasets are available for Reynolds numbers 200, 400, and 600.

### Training Approach
1. **Low Reynolds Training**: Train on Re=200 data
2. **Transfer Learning**: Fine-tune on higher Reynolds numbers (Re=400, Re=600)
3. **Extrapolation Testing**: Evaluate performance on unseen Reynolds numbers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd PINN-Extrapolation-Reynolds-Number

# Install dependencies
pip install torch torchvision torchaudio scipy matplotlib numpy pandas
```

## Usage

### Training Enhanced PINN

```bash
# Train on low Reynolds number
cd enhanced_pinn
python train_lowRe.py

# Train on high Reynolds number with transfer learning
python train_highRe.py
```

### Running Standard PINN

```bash
# Run standard PINN implementation
python standardpinn.py
```

### Running Standard Neural Network Baseline

```bash
# Run standard neural network baseline
python standard_nn.py
```

## Results

The project evaluates model performance using relative L2 error metrics across different Reynolds numbers:

- **Interpolation Zone** (Re ≤ 700): Both PINNs and standard NNs perform well
- **Extrapolation Zone** (Re > 700): PINNs show superior generalization compared to standard NNs
- **Breakdown Point**: PINNs maintain accuracy up to Re ≈ 1000, while standard NNs break down around Re ≈ 700

## Key Findings

1. PINNs demonstrate superior extrapolation capabilities compared to standard neural networks
2. Transfer learning from low to high Reynolds numbers improves training efficiency
3. Physics-informed constraints help maintain accuracy in extrapolation regimes
4. There's a clear breakdown point beyond which both approaches fail, but PINNs extend this limit significantly

## References

This work builds upon the foundational PINNs research:
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

## License

This project is licensed under the MIT License - see the LICENSE file for details.