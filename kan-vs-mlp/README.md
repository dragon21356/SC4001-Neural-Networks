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
- `src/kan_layer.py`: Placeholder for KAN layer implementations.
- `src/bspline_activation.py`: Placeholder for B-spline activation logic.
- `src/models.py`: Placeholder for MLP and KAN model definitions.
- `src/data_utils.py`: Placeholder for dataset loading and preprocessing helpers.
- `src/train.py`: Placeholder for model training routines.
- `src/evaluate.py`: Placeholder for evaluation metrics and benchmarking code.
- `src/spline_vis.py`: Placeholder for spline and activation visualizations.
- `src/utils.py`: Shared utilities for seeding, plotting, CSV I/O, and experiment helpers.
- `experiments/exp1_regression.py`: Placeholder entry point for regression experiments.
- `experiments/exp2_cifar10.py`: Placeholder entry point for CIFAR-10 experiments.
- `experiments/exp3_interpretability.py`: Placeholder entry point for interpretability experiments.
- `experiments/configs/exp1_config.yaml`: Placeholder configuration for Experiment 1.
- `experiments/configs/exp2_config.yaml`: Placeholder configuration for Experiment 2.
- `notebooks/Exp1_Regression.ipynb`: Placeholder notebook for regression analysis.
- `notebooks/Exp2_CIFAR10.ipynb`: Placeholder notebook for CIFAR-10 analysis.
- `notebooks/Exp3_Interpretability.ipynb`: Placeholder notebook for interpretability analysis.
- `notebooks/Results_Visualization.ipynb`: Placeholder notebook for result aggregation and plots.
- `results/exp1/`: Saved outputs for Experiment 1.
- `results/exp2/`: Saved outputs for Experiment 2.
- `results/exp3/`: Saved outputs for Experiment 3.
- `report/figures/`: Figures intended for the final written report.
- `requirements.txt`: Python dependency list for the project.
- `setup_project.sh`: Script to recreate the scaffolded project structure.
- `README.md`: Project overview and usage notes.

## How to Reproduce

Instructions will be added after experiments are complete.

## Citation

The `KANLinear` implementation in this project is adapted from `efficient-kan` by Blealtan and is used under the MIT license:
https://github.com/Blealtan/efficient-kan
