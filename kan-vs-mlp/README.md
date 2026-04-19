# KAN vs MLP: A Systematic Benchmark Across Tasks

This project was developed for the SC4001 Neural Networks and Deep Learning course. It benchmarks Kolmogorov-Arnold Networks (KANs) against standard multilayer perceptrons (MLPs) across three linked studies: low-dimensional regression on California Housing, CNN classification heads on CIFAR-10, and an interpretability comparison between KAN spline nonlinearity and SHAP feature importance. The codebase is organized to support both reproducible experimentation and the final course report.

The repository also includes a novel BSpline-MLP variant. This model keeps the standard MLP weight structure but replaces fixed ReLU activations with learnable B-spline activations, letting us test whether any KAN advantage comes from learnable activation functions alone or from the broader KAN structural redesign.

## Requirements

- Python 3.10+
- Internet connection for the first dataset download
- Optional NVIDIA GPU for faster Experiment 2 training

## Installation

Create and activate a virtual environment from the project root.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows Command Prompt

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Notes on datasets

- `src/data_utils.py` downloads CIFAR-10 automatically with `download=True`.
- `src/data_utils.py` also fetches California Housing automatically into `./data`.
- This means the project can be submitted without the `data/` folder, but the machine reproducing the experiments must have internet access the first time the loaders are run.

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
│   ├── exp1_analysis.py
│   ├── exp2_cifar10.py
│   ├── exp2_analysis.py
│   ├── exp2_preflight.py
│   ├── cifar10_diagnostic.py
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
│   ├── figures/
│   └── report_data.md
├── requirements.txt
├── setup_project.sh
├── integration_test.py
└── README.md
```

- `src/kan_layer.py`: Efficient `KANLinear` implementation adapted from `efficient-kan`.
- `src/bspline_activation.py`: Learnable per-feature B-spline activation used in the BSpline-MLP baseline.
- `src/models.py`: Regression models, CIFAR-10 CNN backbone, and all classification heads.
- `src/data_utils.py`: California Housing and CIFAR-10 dataset loading, preprocessing, and splits.
- `src/train.py`: Generic training loop, validation, scheduler handling, and checkpoint loading.
- `src/evaluate.py`: Accuracy, R2, MSE, convergence epoch, and classification reporting helpers.
- `src/spline_vis.py`: KAN probing and spline visualization utilities.
- `src/utils.py`: Seeding, device selection, plotting helpers, CSV utilities, and formatting.
- `experiments/exp1_regression.py`: Experiment 1 sweep for KAN vs matched MLP on California Housing.
- `experiments/exp1_analysis.py`: Experiment 1 analysis, figures, and spline extraction.
- `experiments/cifar10_diagnostic.py`: Diagnostic script used to choose CIFAR-10 learning rates.
- `experiments/exp2_preflight.py`: Short-run verification before the full CIFAR-10 sweep.
- `experiments/exp2_cifar10.py`: Full Experiment 2 classification sweep.
- `experiments/exp2_analysis.py`: Experiment 2 analysis, figures, and summary tables.
- `experiments/exp3_interpretability.py`: SHAP vs KAN nonlinearity comparison for interpretability.
- `experiments/configs/exp1_config.yaml`: Experiment 1 hyperparameters and search space.
- `experiments/configs/exp2_config.yaml`: Experiment 2 model settings and training configuration.
- `results/exp1`, `results/exp2`, `results/exp3`: Saved experiment outputs and figures.
- `report/report_data.md`: Extracted numerical reference sheet for report writing.
- `integration_test.py`: End-to-end integration test across data, models, training, and utilities.

## How to Reproduce

Run all commands from the `kan-vs-mlp/` project root.

### 1. Optional: Full integration test

```powershell
python integration_test.py
```

### 2. Experiment 1: Regression

Run the full sweep:

```powershell
python experiments\exp1_regression.py
```

Generate Experiment 1 figures and spline outputs:

```powershell
python experiments\exp1_analysis.py
```

### 3. Experiment 2: CIFAR-10 classification

Optional pre-flight safety check:

```powershell
python experiments\exp2_preflight.py
```

Optional learning-rate diagnostic:

```powershell
python experiments\cifar10_diagnostic.py
```

Run the full classification sweep:

```powershell
python experiments\exp2_cifar10.py
```

Generate Experiment 2 figures and tables:

```powershell
python experiments\exp2_analysis.py
```

### 4. Experiment 3: Interpretability

Run the SHAP vs KAN comparison:

```powershell
python experiments\exp3_interpretability.py
```

## Expected Outputs

- Experiment 1 writes results to `results/exp1/`
- Experiment 2 writes results to `results/exp2/`
- Experiment 3 writes results to `results/exp3/`
- Report-ready extracted numbers are stored in `report/report_data.md`

## Citation

The `KANLinear` implementation used in this project is adapted from `efficient-kan` by Blealtan and used under the MIT license:

https://github.com/Blealtan/efficient-kan
