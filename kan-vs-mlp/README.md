# KAN vs MLP: A Systematic Benchmark Across Tasks

This project supports the SC4001 Neural Networks and Deep Learning course by benchmarking Kolmogorov-Arnold Networks (KANs) against standard multilayer perceptrons (MLPs) on regression, image classification, and interpretability-focused experiments. The repository is structured for reproducible deep learning research in PyTorch and is intended to support both the final 10-page report and the submitted code package.

## Installation

Install the project dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
kan-vs-mlp/
├── src/
│   ├── __init__.py
│   ├── kan_layer.py
│   ├── bspline_activation.py
│   ├── models.py
│   ├── data_utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── spline_vis.py
│   └── utils.py
├── experiments/
│   ├── exp1_regression.py
│   ├── exp2_cifar10.py
│   ├── exp3_interpretability.py
│   └── configs/
│       ├── exp1_config.yaml
│       └── exp2_config.yaml
├── notebooks/
│   ├── Exp1_Regression.ipynb
│   ├── Exp2_CIFAR10.ipynb
│   ├── Exp3_Interpretability.ipynb
│   └── Results_Visualization.ipynb
├── results/
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── report/
│   └── figures/
├── requirements.txt
├── setup_project.sh
└── README.md
```

- `src/__init__.py`: Package marker for the source module.
- `src/kan_layer.py`: Efficient `KANLinear` (B-spline + base path).
- `src/bspline_activation.py`: Learnable per-feature B-spline activations (BSpline-MLP head).
- `src/models.py`: MLP/KAN regressors, CNN backbone, and CIFAR-10 heads.
- `src/data_utils.py`: California Housing and CIFAR-10 loaders with splits and transforms.
- `src/train.py`: Training loop, validation, early stopping, checkpoints.
- `src/evaluate.py`: Metrics (R², accuracy), test evaluation, convergence epoch helper.
- `src/spline_vis.py`: Probing and plots for KAN spline interpretability.
- `src/utils.py`: Seeding, plotting, CSV I/O, and experiment helpers.
- `experiments/exp1_regression.py`: Experiment 1 California Housing sweep (KAN vs matched MLP).
- `experiments/exp2_cifar10.py`: Experiment 2 CIFAR-10 CNN + swappable heads.
- `experiments/exp3_interpretability.py`: SHAP vs KAN nonlinearity (requires Exp1 outputs).
- `experiments/configs/exp1_config.yaml`: Experiment 1 hyperparameters and grid.
- `experiments/configs/exp2_config.yaml`: Experiment 2 models and training settings.
- `notebooks/Exp1_Regression.ipynb`: Regression analysis and plots.
- `notebooks/Exp2_CIFAR10.ipynb`: CIFAR-10 analysis and plots.
- `notebooks/Exp3_Interpretability.ipynb`: Interpretability analysis.
- `notebooks/Results_Visualization.ipynb`: Aggregated results and figures.
- `results/exp1/`: Saved outputs for Experiment 1.
- `results/exp2/`: Saved outputs for Experiment 2.
- `results/exp3/`: Saved outputs for Experiment 3.
- `report/figures/`: Figures intended for the final written report.
- `requirements.txt`: Python dependency list for the project.
- `setup_project.sh`: Script to recreate the scaffolded project structure.
- `README.md`: Project overview and usage notes.

## How to Reproduce

Run experiment scripts from `kan-vs-mlp/` with the default configs (e.g. `python experiments/exp1_regression.py`). Use `python integration_test.py` for a full smoke test of data loaders, models, and short training runs.

## Citation

The `KANLinear` implementation in this project is adapted from `efficient-kan` by Blealtan and is used under the MIT license:
https://github.com/Blealtan/efficient-kan
