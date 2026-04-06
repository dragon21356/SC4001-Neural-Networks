# Project Context Brief

## SECTION 1: PROJECT OVERVIEW

This project is a PyTorch research benchmark for `SC4001 Neural Networks and Deep Learning` comparing `Kolmogorov-Arnold Networks (KANs)` against standard `MLPs` across regression, image classification, and interpretability analysis. The repository is structured as a coursework submission with reproducible experiment scripts, saved CSV outputs, generated report figures, and placeholder notebooks/report folders. The KAN implementation is adapted from `efficient-kan` by Blealtan (MIT license) and exposed through `src/kan_layer.py` as `KANLinear`, which acts as a drop-in structural alternative to `nn.Linear`.

The project is organized around three research questions. `Experiment 1` asks whether KAN outperforms a parameter-matched MLP on low-dimensional tabular regression using California Housing. `Experiment 2` asks whether replacing the classification head in a shared CNN backbone with KAN-based or spline-based alternatives improves CIFAR-10 performance, and whether gains come from structure or activations. `Experiment 3` asks whether KAN’s learned spline functions provide useful interpretability by comparing KAN feature nonlinearity rankings against SHAP values from an MLP. The novel `BSpline-MLP` variant uses standard `nn.Linear` layers but replaces `ReLU` with a learnable `BSplineActivation`, testing the specific hypothesis that KAN’s advantage may come from learnable activation functions rather than the full edge-function redesign.

## SECTION 2: COMPLETE FILE STRUCTURE

```text
kan-vs-mlp/
├── PROJECT_CONTEXT_BRIEF.md                # This saved context-transfer brief
├── README.md                               # Project overview, structure, install instructions, citation note
├── requirements.txt                        # Python package requirements with major-version pins
├── setup_project.sh                        # Shell scaffold script that creates the project layout
├── integration_test.py                     # 18-test end-to-end integration harness for Phase 0
├── data/
│   ├── cal_housing_py3.pkz                 # Cached California Housing dataset from scikit-learn
│   ├── cifar-10-python.tar.gz              # Downloaded CIFAR-10 archive
│   └── cifar-10-batches-py/
│       ├── batches.meta                    # CIFAR-10 metadata
│       ├── data_batch_1                    # CIFAR-10 train batch 1
│       ├── data_batch_2                    # CIFAR-10 train batch 2
│       ├── data_batch_3                    # CIFAR-10 train batch 3
│       ├── data_batch_4                    # CIFAR-10 train batch 4
│       ├── data_batch_5                    # CIFAR-10 train batch 5
│       ├── readme.html                     # CIFAR-10 bundled readme
│       └── test_batch                      # CIFAR-10 test batch
├── experiments/
│   ├── cifar10_diagnostic.py               # Phase 1 diagnostic for CIFAR-10 pipeline validation and LR sweeps
│   ├── exp1_analysis.py                    # Post-run analysis and figure generation for Experiment 1
│   ├── exp1_regression.py                  # Full Experiment 1 regression sweep across KAN and matched MLP configs
│   ├── exp2_cifar10.py                     # Empty placeholder for Phase 2 full CIFAR-10 sweep
│   ├── exp3_interpretability.py            # Empty placeholder for Phase 3 SHAP vs spline analysis
│   ├── configs/
│   │   ├── exp1_config.yaml                # Experiment 1 hyperparameter config
│   │   └── exp2_config.yaml                # Empty placeholder config for Experiment 2
│   └── __pycache__/                        # Python bytecode cache
├── notebooks/
│   ├── Exp1_Regression.ipynb               # Placeholder notebook
│   ├── Exp2_CIFAR10.ipynb                  # Placeholder notebook
│   ├── Exp3_Interpretability.ipynb         # Placeholder notebook
│   └── Results_Visualization.ipynb         # Placeholder notebook
├── report/
│   └── figures/                            # Report figure export directory
├── results/
│   ├── exp1/
│   │   ├── exp1_all_results.csv            # Run-level Experiment 1 results, 48 rows
│   │   ├── exp1_summary.csv                # Seed-averaged Experiment 1 summary, 16 rows
│   │   ├── feature_names.json              # Saved California Housing feature names
│   │   ├── kan_nonlinearity_scores.npy     # Saved KAN feature nonlinearity scores
│   │   ├── checkpoints/                    # Saved best checkpoints for every Experiment 1 run
│   │   └── figures/
│   │       ├── fig1_kan_r2_heatmap.png     # KAN R² heatmap over grid size and spline order
│   │       ├── fig2_kan_vs_mlp_r2.png      # Grouped bar chart for KAN vs matched MLP test R²
│   │       ├── fig3_convergence_curves.png # Best KAN vs matched MLP convergence curves
│   │       ├── fig4_mse_comparison.png     # Grouped bar chart for test MSE
│   │       ├── fig5_training_speed.png     # Training-speed comparison figure
│   │       ├── fig6_params_vs_r2.png       # Parameter efficiency scatter plot
│   │       ├── fig7_kan_splines.png        # Learned spline responses from best KAN first layer
│   │       ├── fig7b_feature_importance_proxy.png # Proxy comparison plot generated before SHAP is built
│   │       ├── fig7c_spline_vs_proxy_scatter.png  # Proxy scatter plot generated before SHAP is built
│   │       └── summary_table.txt           # Text table for report copy-paste
│   ├── exp2/
│   │   └── diagnostic/
│   │       ├── kan_lr_sweep.png            # KAN LR sweep figure for Mode A and Mode B
│   │       ├── lr_sweep_results.csv        # Saved LR sweep numeric results
│   │       ├── pipeline_validation_curves.png # 10-epoch pipeline validation curves
│   │       └── recommended_hyperparams.yaml # Saved Phase 2 hyperparameter recommendations
│   └── exp3/
│       ├── test_comparison.png             # Smoke-test artifact from spline_vis.py
│       ├── test_scatter.png                # Smoke-test artifact from spline_vis.py
│       └── test_splines.png                # Smoke-test artifact from spline_vis.py
└── src/
    ├── __init__.py                         # Empty package marker
    ├── bspline_activation.py               # Learnable feature-wise B-spline activation module
    ├── data_utils.py                       # California Housing and CIFAR-10 loaders
    ├── evaluate.py                         # Evaluation metrics and test helpers
    ├── kan_layer.py                        # Efficient KANLinear implementation
    ├── models.py                           # Regression models, CNN backbone, and classification heads
    ├── spline_vis.py                       # KAN spline probing, reconstruction, and plotting utilities
    ├── train.py                            # Generic training loop and EarlyStopping
    ├── utils.py                            # Shared plotting, device, seed, CSV, and formatting helpers
    └── __pycache__/                        # Python bytecode cache
```

## SECTION 3: MODULE API REFERENCE

### `src/utils.py`
- `set_seed(seed: int = 42) -> None` — Sets Python, NumPy, and PyTorch seeds and configures deterministic CuDNN behavior.
- `get_device() -> torch.device` — Returns `cuda` when available, otherwise `cpu`, and prints the choice.
- `plot_training_curves(history: dict, title: str, save_path: str | None = None) -> None` — Plots training loss, validation loss, and optional validation metric.
- `plot_bar_comparison(labels: list[str], values: list[float], errors: list[float] | None = None, ylabel: str = '', title: str = '', colors: list[str] | None = None, save_path: str | None = None) -> None` — Plots a styled comparison bar chart.
- `save_results_csv(results: dict, filepath: str) -> None` — Saves a column-oriented result dictionary to CSV.
- `load_results_csv(filepath: str) -> pd.DataFrame` — Loads a CSV file into a DataFrame.
- `format_param_count(n: int) -> str` — Formats a parameter count such as `133898` as `133.9K`.
- `MLP_COLOR = "#2196F3"` — Shared blue color for MLP plots.
- `KAN_COLOR = "#FF9800"` — Shared orange color for KAN plots.
- `BSPLINE_COLOR = "#4CAF50"` — Shared green color for BSpline-MLP plots.

### `src/kan_layer.py`
- `KANLinear(in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, scale_noise: float = 0.1, scale_base: float = 1.0, scale_spline: float = 1.0, enable_standalone_scale_spline: bool = True, base_activation: type[nn.Module] = nn.SiLU, grid_eps: float = 0.02, grid_range: Sequence[float] = (-1.0, 1.0)) -> None` — Efficient KAN edge-function layer with base and spline paths.
- `KANLinear.reset_parameters(self) -> None` — Initializes base and spline parameters.
- `KANLinear.b_splines(self, x: torch.Tensor) -> torch.Tensor` — Computes Cox-de Boor B-spline basis values.
- `KANLinear.forward(self, x: torch.Tensor) -> torch.Tensor` — Returns `base_output + spline_output`.
- `KANLinear.update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None` — Adapts the knot grid to data quantiles and refits coefficients.
- `KANLinear.scaled_spline_weight` — Property returning `spline_weight` with optional learned scaling applied.

### `src/bspline_activation.py`
- `BSplineActivation(num_features: int, grid_size: int = 5, spline_order: int = 3, grid_range: tuple[float, float] = (-2.0, 2.0)) -> None` — Feature-wise learnable spline activation with LayerNorm + SiLU residual path.
- `BSplineActivation.reset_parameters(self) -> None` — Initializes spline coefficients, base weights, and LayerNorm.
- `BSplineActivation.b_splines(self, x: torch.Tensor) -> torch.Tensor` — Computes B-spline basis values per feature.
- `BSplineActivation.forward(self, x: torch.Tensor) -> torch.Tensor` — Applies the learnable activation and returns the same shape as input.

### `src/models.py`
- `MLPRegressor(in_features: int = 8, hidden_dim: int = 64, out_features: int = 1, dropout: float = 0.1) -> None` — Tabular regression MLP.
- `KANRegressor(in_features: int = 8, hidden_dim: int = 64, out_features: int = 1, grid_size: int = 5, spline_order: int = 3) -> None` — Tabular regression KAN.
- `CNNBackbone() -> None` — Four-block CIFAR-10 feature extractor that outputs 512 features.
- `MLPHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, dropout: float = 0.3) -> None` — Standard classification head.
- `KANHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, grid_size: int = 5, spline_order: int = 3) -> None` — Two-layer KAN classification head.
- `BSplineMLPHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, dropout: float = 0.3, grid_size: int = 5, spline_order: int = 3) -> None` — Standard linear head with `BSplineActivation` in place of `ReLU`.
- `CNNWithHead(backbone: nn.Module, head: nn.Module) -> None` — Wrapper that composes a backbone and head.
- `build_cifar10_model(head_type: str, hidden_dim: int = 256, grid_size: int = 5, spline_order: int = 3, dropout: float = 0.3) -> CNNWithHead` — Builds a fresh `CNNBackbone` plus requested head type.
- `count_parameters(model: nn.Module) -> int` — Counts trainable parameters in a model.
- `count_head_parameters(model: CNNWithHead) -> int` — Counts trainable parameters in `model.head`.
- `model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> None` — Prints a model summary and verifies a forward pass.

### `src/data_utils.py`
- `get_california_housing(batch_size: int = 64, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]` — Loads, splits, scales, tensors, and returns California Housing loaders.
- `get_cifar10(batch_size: int = 128, val_ratio: float = 0.1, seed: int = 42, data_dir: str = './data', num_workers: int = 2) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]` — Loads CIFAR-10 with augmented train split and non-augmented validation split.
- `get_dataset(name: str, **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]` — Dispatches to the supported dataset loader by name.

### `src/train.py`
- `EarlyStopping(patience: int = 10, min_delta: float = 0.0001, mode: str = 'min') -> None` — Tracks validation-loss improvements and signals stopping.
- `train_one_epoch(model: nn.Module, train_loader: Any, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> dict[str, float]` — Runs one training epoch and returns average train loss.
- `validate(model: nn.Module, val_loader: Any, criterion: nn.Module, device: torch.device, metric_fn: MetricFn | None = None) -> dict[str, float]` — Runs evaluation on a validation-style loader.
- `train_model(model: nn.Module, train_loader: Any, val_loader: Any, optimizer: torch.optim.Optimizer, criterion: nn.Module, epochs: int, device: torch.device, metric_fn: MetricFn | None = None, scheduler: Any | None = None, early_stopping_patience: int | None = 10, checkpoint_dir: str | None = None, model_name: str = 'model', verbose: bool = True) -> dict[str, Any]` — Generic task-agnostic training loop with checkpointing, scheduler handling, and optional early stopping.

### `src/evaluate.py`
- `accuracy_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float` — Returns classification accuracy from raw logits.
- `r2_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float` — Returns regression R².
- `mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float` — Returns mean squared error.
- `evaluate_model(model: nn.Module, test_loader: Any, criterion: nn.Module, device: torch.device, metric_fn: MetricFn | None = None) -> dict[str, float]` — Evaluates a trained model on a test loader.
- `per_class_accuracy(model: nn.Module, test_loader: Any, device: torch.device, class_names: list[str] | None = None) -> dict[str | int, float]` — Computes per-class accuracy for a classifier.
- `get_classification_report(model: nn.Module, test_loader: Any, device: torch.device, class_names: list[str] | None = None) -> str` — Returns a scikit-learn classification report string.
- `compute_convergence_epoch(history: dict[str, Any], threshold_fraction: float = 0.95) -> int` — Returns the first epoch that reaches 95% of final metric quality.

### `src/spline_vis.py`
- `probe_kan_layer_responses(layer: KANLinear, input_range: tuple[float, float] = (-3.0, 3.0), num_points: int = 200, reference_value: float = 0.0, device: str = 'cpu') -> dict[str, Any]` — Sweeps one feature at a time and records effective layer responses.
- `reconstruct_bsplines(layer: KANLinear) -> dict[str, Any]` — Reconstructs spline functions directly from knot vectors and coefficients using SciPy.
- `compute_feature_nonlinearity(responses: dict[str, Any]) -> np.ndarray` — Computes per-feature nonlinearity scores from probed responses.
- `plot_kan_splines(responses: dict[str, Any], feature_names: list[str] | None = None, title: str = 'Learned KAN Activation Functions', save_path: str | None = None, figsize: tuple[float, float] | None = None) -> None` — Plots probed KAN response curves by feature.
- `plot_feature_importance_comparison(kan_nonlinearity: np.ndarray, shap_importance: np.ndarray, feature_names: list[str], title: str = 'Feature Importance: KAN Nonlinearity vs SHAP', save_path: str | None = None) -> None` — Creates side-by-side importance ranking plots.
- `plot_spline_vs_shap_scatter(kan_nonlinearity: np.ndarray, shap_importance: np.ndarray, feature_names: list[str], title: str = 'KAN Nonlinearity vs SHAP Importance', save_path: str | None = None) -> None` — Creates a scatter plot with Spearman correlation.

## SECTION 4: MODEL ARCHITECTURES (EXACT)

### `MLPRegressor`
- Constructor: `MLPRegressor(in_features: int = 8, hidden_dim: int = 64, out_features: int = 1, dropout: float = 0.1)`
- Exact layer sequence:
  `Linear(in_features, hidden_dim) -> ReLU() -> Dropout(dropout) -> Linear(hidden_dim, hidden_dim) -> ReLU() -> Dropout(dropout) -> Linear(hidden_dim, out_features)`
- Forward input/output:
  `(batch, in_features) -> (batch, out_features)`
- Parameter formula from the actual implementation:
  `(in_features * H + H) + (H * H + H) + (H * out_features + out_features)`
- California Housing default formula:
  `H^2 + 11H + 1`
- Default parameter count with `H=64`:
  `4801`

### `KANRegressor`
- Constructor: `KANRegressor(in_features: int = 8, hidden_dim: int = 64, out_features: int = 1, grid_size: int = 5, spline_order: int = 3)`
- Exact layer sequence:
  `KANLinear(in_features, hidden_dim, grid_size, spline_order) -> KANLinear(hidden_dim, out_features, grid_size, spline_order)`
- Forward input/output:
  `(batch, in_features) -> (batch, out_features)`
- Parameter formula:
  `in_features * H * (G + k + 2) + H * out_features * (G + k + 2)`
- Default parameter count with `H=64, G=5, k=3`:
  `5760`

### `CNNBackbone`
- Constructor: `CNNBackbone()`
- Exact layer sequence:
  `Conv2d(3,64,3,padding=1) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(2) -> Conv2d(64,128,3,padding=1) -> BatchNorm2d(128) -> ReLU() -> MaxPool2d(2) -> Conv2d(128,256,3,padding=1) -> BatchNorm2d(256) -> ReLU() -> MaxPool2d(2) -> Conv2d(256,512,3,padding=1) -> BatchNorm2d(512) -> ReLU() -> AdaptiveAvgPool2d(1) -> Flatten()`
- Forward input/output:
  `(batch, 3, 32, 32) -> (batch, 512)`
- `output_dim` property:
  returns `512`
- Weight initialization:
  `kaiming_normal_` for conv weights, `1/0` constants for batch norm weight/bias
- Parameter count:
  `1,552,896`

### `MLPHead`
- Constructor: `MLPHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, dropout: float = 0.3)`
- Exact layer sequence:
  `Linear(in_features, hidden_dim) -> ReLU() -> Dropout(dropout) -> Linear(hidden_dim, n_classes)`
- Forward input/output:
  `(batch, 512) -> (batch, 10)`
- Parameter formula:
  `(in_features * H + H) + (H * n_classes + n_classes)`
- Default parameter count with `H=256`:
  `133,898`

### `KANHead`
- Constructor: `KANHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, grid_size: int = 5, spline_order: int = 3)`
- Exact layer sequence:
  `KANLinear(in_features, hidden_dim, grid_size, spline_order) -> KANLinear(hidden_dim, n_classes, grid_size, spline_order)`
- Forward input/output:
  `(batch, 512) -> (batch, 10)`
- Parameter formula:
  `in_features * H * (G + k + 2) + H * n_classes * (G + k + 2)`
- Mode A parameter count with `H=256, G=5, k=3`:
  `1,336,320`
- Mode B parameter count with `H=28, G=5, k=3`:
  `146,160`

### `BSplineMLPHead`
- Constructor: `BSplineMLPHead(in_features: int = 512, hidden_dim: int = 256, n_classes: int = 10, dropout: float = 0.3, grid_size: int = 5, spline_order: int = 3)`
- Exact layer sequence:
  `Linear(in_features, hidden_dim) -> BSplineActivation(hidden_dim, grid_size, spline_order) -> Dropout(dropout) -> Linear(hidden_dim, n_classes)`
- Forward input/output:
  `(batch, 512) -> (batch, 10)`
- Parameter formula:
  linear params from `MLPHead` plus `BSplineActivation` params, where `BSplineActivation` contributes `H * (G + k) + H + 2H`
- Default parameter count with `H=256, G=5, k=3`:
  `136,714`

### `CNNWithHead`
- Constructor: `CNNWithHead(backbone: nn.Module, head: nn.Module)`
- Exact forward path:
  `features = backbone(x)` then `head(features)`
- Forward input/output:
  `(batch, 3, 32, 32) -> (batch, 10)`
- Parameter formula:
  `count_parameters(backbone) + count_parameters(head)`

### `build_cifar10_model`
- Signature: `build_cifar10_model(head_type: str, hidden_dim: int = 256, grid_size: int = 5, spline_order: int = 3, dropout: float = 0.3) -> CNNWithHead`
- Behavior:
  creates a fresh `CNNBackbone()` and one of `MLPHead`, `KANHead`, or `BSplineMLPHead`, then returns `CNNWithHead(backbone, head)`
- Default total parameter counts:
  - `build_cifar10_model("mlp", hidden_dim=256)` -> `1,686,794`
  - `build_cifar10_model("kan", hidden_dim=256)` -> `2,889,216`
  - `build_cifar10_model("kan", hidden_dim=28)` -> `1,699,056`
  - `build_cifar10_model("bspline_mlp", hidden_dim=256)` -> `1,689,610`

## SECTION 5: EXPERIMENT 1 RESULTS (COMPLETED)

`Experiment 1` has been executed. The run-level CSV exists at [results/exp1/exp1_all_results.csv](/Users/vivsu/Desktop/SC4001-Neural-Networks/kan-vs-mlp/results/exp1/exp1_all_results.csv) and contains `48` rows, matching `8 configs x 3 seeds x 2 model types`. The seed-averaged summary exists at [results/exp1/exp1_summary.csv](/Users/vivsu/Desktop/SC4001-Neural-Networks/kan-vs-mlp/results/exp1/exp1_summary.csv) and contains `16` rows.

Full summary table transcribed from `exp1_summary.csv`:

| model_type | config_name | grid_size | spline_order | hidden_dim | total_params | test_mse_mean | test_mse_std | test_r2_mean | test_r2_std | convergence_epoch_mean | convergence_epoch_std | avg_time_per_epoch_mean | total_training_time_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kan | G10_K2 | 10 | 2 | 64 | 8064 | 0.2530574305 | 0.0046823969 | 0.8127013048 | 0.0019525287 | 5.0 | 1.7320508076 | 3.2772801161 | 112.2619727453 |
| kan | G10_K3 | 10 | 3 | 64 | 8640 | 0.2431744461 | 0.0101041904 | 0.8198518554 | 0.0117250336 | 5.0 | 0.0 | 4.5280768078 | 164.3664178848 |
| kan | G20_K2 | 20 | 2 | 64 | 13824 | 0.2665392597 | 0.0082880903 | 0.8025528590 | 0.0114401079 | 3.0 | 0.0 | 3.5909028963 | 101.6832055251 |
| kan | G20_K3 | 20 | 3 | 64 | 14400 | 0.2627322201 | 0.0071267348 | 0.8054285844 | 0.0091612879 | 3.0 | 0.0 | 4.5300127923 | 123.8219992320 |
| kan | G3_K2 | 3 | 2 | 64 | 4032 | 0.2606731561 | 0.0064265174 | 0.8069203893 | 0.0099305420 | 9.3333333333 | 0.5773502692 | 1.7557539640 | 137.8497543335 |
| kan | G3_K3 | 3 | 3 | 64 | 4608 | 0.2579135202 | 0.0115618068 | 0.8089100321 | 0.0133798365 | 8.6666666667 | 2.0816659995 | 3.6548607479 | 296.9279492696 |
| kan | G5_K2 | 5 | 2 | 64 | 5184 | 0.2513333794 | 0.0054247969 | 0.8138552705 | 0.0087610175 | 7.0 | 0.0 | 3.2633855395 | 196.5083043575 |
| kan | G5_K3 | 5 | 3 | 64 | 5760 | 0.2554871688 | 0.0054464746 | 0.8107719421 | 0.0091388540 | 7.0 | 1.0 | 4.2011771063 | 276.2515815099 |
| mlp | G10_K2_matched | NaN | NaN | 84 | 7981 | 0.2489165611 | 0.0095730491 | 0.8155838251 | 0.0120772475 | 18.3333333333 | 2.5166114784 | 1.0389013993 | 164.6671978633 |
| mlp | G10_K3_matched | NaN | NaN | 88 | 8713 | 0.2490422078 | 0.0068503449 | 0.8155245582 | 0.0100686460 | 17.3333333333 | 5.0332229568 | 1.0136353988 | 148.8041424751 |
| mlp | G20_K2_matched | NaN | NaN | 112 | 13777 | 0.2486818286 | 0.0138522733 | 0.8157033324 | 0.0152605540 | 15.6666666667 | 5.0332229568 | 1.0628986099 | 136.3248944283 |
| mlp | G20_K3_matched | NaN | NaN | 115 | 14491 | 0.2502066817 | 0.0060898990 | 0.8146739006 | 0.0094923773 | 14.0 | 2.0 | 1.0969983980 | 131.4613229434 |
| mlp | G3_K2_matched | NaN | NaN | 58 | 4003 | 0.2599545807 | 0.0079629044 | 0.8074478706 | 0.0106037031 | 21.0 | 1.0 | 0.5395775805 | 58.1101926168 |
| mlp | G3_K3_matched | NaN | NaN | 63 | 4663 | 0.2544931827 | 0.0039172033 | 0.8115947247 | 0.0049658027 | 19.3333333333 | 2.8867513459 | 1.0151279185 | 140.3983486493 |
| mlp | G5_K2_matched | NaN | NaN | 67 | 5227 | 0.2466801792 | 0.0087392330 | 0.8172561526 | 0.0112681920 | 21.0 | 4.3588989435 | 1.1173727982 | 182.5621612072 |
| mlp | G5_K3_matched | NaN | NaN | 71 | 5823 | 0.2506547134 | 0.0070716585 | 0.8143509030 | 0.0096143901 | 18.6666666667 | 2.0816659995 | 1.0055579782 | 159.0949476560 |

Key findings from the actual saved CSVs:
- Best KAN summary config: `G10_K3`
  - `test_r2_mean = 0.8198518554`
  - `test_r2_std = 0.0117250336`
  - `test_mse_mean = 0.2431744461`
  - `avg_time_per_epoch_mean = 4.5280768078s`
- Best matched MLP summary config: `G5_K2_matched`
  - `hidden_dim = 67`
  - `test_r2_mean = 0.8172561526`
  - `test_r2_std = 0.0112681920`
  - `test_mse_mean = 0.2466801792`
  - `avg_time_per_epoch_mean = 1.1173727982s`
- Winner by mean test R²:
  `KAN` by a margin of `0.0025957028`
- Best individual KAN run:
  - `config_name = G10_K3`
  - `seed = 456`
  - `test_r2 = 0.8316593766`
  - checkpoint: `results/exp1/checkpoints/kan_G10_K3_seed456_best.pt`
- Best individual MLP run:
  - `config_name = G20_K2_matched`
  - `seed = 123`
  - `test_r2 = 0.8246796727`
  - checkpoint: `results/exp1/checkpoints/mlp_G20_K2_matched_seed123_best.pt`

Feature nonlinearity ranking loaded from `results/exp1/kan_nonlinearity_scores.npy` and `results/exp1/feature_names.json`:
1. `AveOccup` — `0.0258230133`
2. `Population` — `0.0240338570`
3. `Latitude` — `0.0206145252`
4. `HouseAge` — `0.0166963949`
5. `Longitude` — `0.0120597284`
6. `AveBedrms` — `0.0097664528`
7. `MedInc` — `0.0065714786`
8. `AveRooms` — `0.0055109400`

Generated Experiment 1 figures saved under `results/exp1/figures/`:
- `fig1_kan_r2_heatmap.png`
- `fig2_kan_vs_mlp_r2.png`
- `fig3_convergence_curves.png`
- `fig4_mse_comparison.png`
- `fig5_training_speed.png`
- `fig6_params_vs_r2.png`
- `fig7_kan_splines.png`
- `fig7b_feature_importance_proxy.png`
- `fig7c_spline_vs_proxy_scatter.png`
- `summary_table.txt`

## SECTION 6: CIFAR-10 DIAGNOSTIC RESULTS (COMPLETED)

The diagnostic recommendation file exists at `results/exp2/diagnostic/recommended_hyperparams.yaml` and contains:

```yaml
mlp_head:
  hidden_dim: 256
  learning_rate: 0.001
  dropout: 0.3
kan_head_mode_a:
  hidden_dim: 256
  learning_rate: 0.0005
  grid_size: 5
  spline_order: 3
kan_head_mode_b:
  hidden_dim: 28
  learning_rate: 0.002
  grid_size: 5
  spline_order: 3
bspline_mlp_head:
  hidden_dim: 256
  learning_rate: 0.002
  dropout: 0.3
  grid_size: 5
  spline_order: 3
training:
  epochs: 50
  batch_size: 128
  scheduler: cosine_annealing
  early_stopping_patience: 10
  seeds:
  - 42
  - 123
  - 456
```

Exact numeric LR sweep results saved in `results/exp2/diagnostic/lr_sweep_results.csv`:

Mode B (`hidden_dim=28`, 15 epochs):
- `2e-3` -> `final_val_acc = 0.8342`, `final_train_loss = 0.3385`, `status = ok`
- `1e-3` -> `final_val_acc = 0.8272`, `final_train_loss = 0.3789`, `status = ok`
- `5e-4` -> `final_val_acc = 0.8096`, `final_train_loss = 0.4304`, `status = ok`
- `3e-4` -> `final_val_acc = 0.7702`, `final_train_loss = 0.4814`, `status = ok`
- `1e-4` -> `final_val_acc = 0.7538`, `final_train_loss = 0.6499`, `status = ok`
- `5e-5` -> `final_val_acc = 0.7086`, `final_train_loss = 0.7985`, `status = ok`
- Recommended LR from the saved YAML and script logic: `2e-3`

Mode A (`hidden_dim=256`, 10 epochs):
- `1e-3` -> `final_val_acc = 0.7620`, `final_train_loss = 0.4808`, `status = ok`
- `5e-4` -> `final_val_acc = 0.7660`, `final_train_loss = 0.5261`, `status = ok`
- `3e-4` -> `final_val_acc = 0.7334`, `final_train_loss = 0.5720`, `status = ok`
- `1e-4` -> `final_val_acc = 0.7268`, `final_train_loss = 0.7197`, `status = ok`
- Recommended LR from the saved YAML and script logic: `5e-4`

Saved pipeline-validation artifacts:
- `results/exp2/diagnostic/pipeline_validation_curves.png`
- `results/exp2/diagnostic/kan_lr_sweep.png`
- `results/exp2/diagnostic/lr_sweep_results.csv`
- `results/exp2/diagnostic/recommended_hyperparams.yaml`

Important evidence note:
- The `10-epoch pipeline validation accuracies and time-per-epoch values for MLPHead / KANHead ModeB / BSplineMLPHead were printed by the diagnostic script but were not persisted to a CSV or YAML file`.
- Because this brief is constrained to actual saved files, the exact file-backed values for those three Part 1 runs cannot be reconstructed numerically from disk alone.
- What is file-backed is that the pipeline-validation figure exists and the later LR sweeps converged successfully for all tested settings.
- Prompt-level context from earlier development said the 10-epoch diagnostic slightly favored `BSplineMLPHead` over `MLPHead`, but that statement is not backed by a saved artifact in the repository.

Key diagnostic takeaway from the code and saved files:
- All tested CIFAR-10 heads trained stably enough to produce LR recommendations.
- `KANHead Mode B` prefers a higher LR (`2e-3`) than `MLPHead` (`1e-3`) and `KANHead Mode A` (`5e-4`).
- `BSplineMLPHead` was assigned `2e-3`, indicating the learnable activation variant tolerated a slightly more aggressive optimizer setting than plain MLP in the diagnostic workflow.

## SECTION 7: EXPERIMENT SCRIPTS STATUS

- `experiments/exp1_regression.py`
  - Exists: `yes`
  - Status: `complete`
  - What it does: loads `exp1_config.yaml`, computes parameter-matched MLP widths, runs all `8 x 3 x 2 = 48` regression runs, supports resume from partial CSV, saves run-level and summary CSVs, and reports best checkpoints.
  - Executed: `yes`
  - Results saved: `yes`, under `results/exp1/`

- `experiments/exp1_analysis.py`
  - Exists: `yes`
  - Status: `complete`
  - What it does: loads Experiment 1 CSVs, generates all report figures, writes the report-ready summary table, probes the best KAN first layer, saves nonlinearity scores and feature names, and attempts spline reconstruction validation.
  - Executed: `yes`
  - Results saved: `yes`, under `results/exp1/figures/` plus `kan_nonlinearity_scores.npy`

- `experiments/cifar10_diagnostic.py`
  - Exists: `yes`
  - Status: `complete`
  - What it does: runs Part 1 pipeline validation, Part 2 KAN LR sweeps, Part 3 MLP/BSpline LR checks, and writes `recommended_hyperparams.yaml`.
  - Executed: `yes`
  - Results saved: `yes`, under `results/exp2/diagnostic/`

- `experiments/exp2_cifar10.py`
  - Exists: `yes`
  - Status: `not implemented`
  - What it does: currently nothing; file is an empty placeholder
  - Executed: `no`
  - Results saved: `no`

- `experiments/exp3_interpretability.py`
  - Exists: `yes`
  - Status: `not implemented`
  - What it does: currently nothing; file is an empty placeholder
  - Executed: `no`
  - Results saved: `no`

## SECTION 8: WHAT REMAINS TO BE DONE

### Phase 2: Experiment 2 — CIFAR-10 Classification (NOT YET BUILT)
- Need to create:
  - `experiments/exp2_cifar10.py`
  - `experiments/exp2_analysis.py`
- Models to compare:
  - `CNN + MLPHead`
  - `CNN + KANHead`
  - `CNN + BSplineMLPHead`
- Two comparison modes planned in the prompt:
  - `Mode A`: same hidden width, `hidden_dim=256` for all heads
  - `Mode B`: parameter-matched KAN head using `hidden_dim=28`, while MLP and BSpline-MLP remain at `hidden_dim=256`
- Learning rates should come from `results/exp2/diagnostic/recommended_hyperparams.yaml`:
  - `MLPHead`: `1e-3`
  - `KANHead Mode A`: `5e-4`
  - `KANHead Mode B`: `2e-3`
  - `BSplineMLPHead`: `2e-3`
- Training config from the saved diagnostic:
  - `epochs=50`
  - `batch_size=128`
  - `scheduler="cosine_annealing"`
  - `early_stopping_patience=10`
  - `seeds=[42, 123, 456]`
- Planned outputs:
  - accuracy tables
  - convergence curves
  - per-class accuracy
  - training-time comparison
- Important planning discrepancy:
  - one project prompt described `18 runs`
  - the implemented diagnostic script’s own final estimate assumes `12 runs` because it counts `MLP`, `KAN Mode A`, `KAN Mode B`, and `BSpline` across `3` seeds
  - the next chat should resolve and lock this experiment design explicitly before implementation

### Phase 3: Experiment 3 — Interpretability (NOT YET BUILT)
- Need to create:
  - `experiments/exp3_interpretability.py`
- Planned workflow from prior prompts:
  - load best KAN regression model from Experiment 1
  - load best MLP regression model from Experiment 1
  - run SHAP `KernelExplainer` on the MLP using `200` test samples
  - compare SHAP feature-importance ranking against KAN nonlinearity ranking already saved at `results/exp1/kan_nonlinearity_scores.npy`
  - generate:
    - side-by-side comparison plot
    - scatter plot with Spearman correlation
- Already available for Experiment 3:
  - `results/exp1/kan_nonlinearity_scores.npy`
  - `results/exp1/feature_names.json`
  - best KAN checkpoint path from Experiment 1
  - best MLP checkpoint path from Experiment 1

### Phase 4: Report Writing
- The report spec from the project prompts is:
  - `10 pages`
  - `Arial 10pt`
  - `A4`
- Planned section allocation from the prompt:
  - Introduction: `1.5 pages`
  - Literature Review: `2 pages`
  - Methodology: `2 pages`
  - Results: `3 pages`
  - Discussion: `1.5 pages`
- File-backed narrative points already available:
  - Best KAN regression config is `G10_K3` with `mean R² = 0.8199`
  - Best matched MLP summary config is `G5_K2_matched` with `mean R² = 0.8173`
  - KAN converged in about `5` epochs to `95%` of final validation quality for its best config, while the best matched MLP summary required about `21` epochs
  - KAN regression was around `4.53s/epoch` for the best config versus `1.12s/epoch` for the best matched MLP summary
- Prompt-level claims such as “R² ~0.82”, “G=20 overfits”, or “BSpline slightly outperformed MLP at 10 epochs” should be treated carefully unless re-verified from saved artifacts.

## SECTION 9: TEAM STRUCTURE AND TIMELINE

The team structure in the prompt was:
- `Person A`: KAN specialist
- `Person B`: CNN / evaluation lead
- `Person C`: report lead

Deadline context from the prompt:
- Deadline: `April 19, 2026, 11:59 PM`

Current date context for this handoff:
- Actual environment date in this session: `April 6, 2026`
- Approximate remaining time to deadline from the actual environment date: `13 days`

Runtime planning context:
- The diagnostic script estimated roughly:
  - `~45-90 minutes on GPU`
  - `~3-5 hours on CPU`
- The prompt-level expectation for Phase 2 was about:
  - `~18 hours on CPU`
  - `~4-5 hours on GPU`
- The exact total for Phase 2 still depends on whether the team runs the `12-run` or `18-run` design.

## SECTION 10: KEY DESIGN DECISIONS AND CONSTRAINTS

1. `KANLinear` in `src/kan_layer.py` is adapted from `efficient-kan` by Blealtan, with the attribution note preserved in both README and source comments.
2. `BSplineActivation` in `src/bspline_activation.py` is a novel module in this project: a drop-in replacement for `ReLU` that learns one spline per feature and does not mix features.
3. The project explicitly compares two possible sources of KAN advantage:
   - learnable activation functions
   - the full edge-function structural redesign
4. Experiment 1 uses parameter matching between KAN and MLP rather than equal hidden width.
5. The true implemented `MLPRegressor` parameter formula is `H^2 + 11H + 1`, not the earlier estimate `H^2 + 10H + 1`; the script detects the mismatch and uses the actual architecture.
6. `FLOPs` were not implemented as a measurement axis. The codebase uses `wall-clock time per epoch` and `parameter count` instead.
7. All classification models output raw logits and all regression models output raw scalars; no `softmax` is applied in model `forward()` methods.
8. The CIFAR-10 backbone is trained from scratch with no pretrained weights, consistent with the course requirement.
9. Early stopping differs by experiment family in the prompt/config layer:
   - Experiment 1 regression config uses `patience=20`
   - Phase 2 classification recommendation uses `patience=10`
10. Shared color scheme across utilities and figures:
    - `MLP = "#2196F3"`
    - `KAN = "#FF9800"`
    - `BSpline = "#4CAF50"`
11. The primary KAN interpretability method in the current code is the `probe_kan_layer_responses(...)` functional probing approach; direct B-spline reconstruction is included as a secondary validation tool.
12. `get_cifar10(...)` deliberately loads the training set twice so training can use augmentation while validation remains augmentation-free on identical split indices.
13. `get_california_housing(...)` fits `StandardScaler` on the training split only, avoiding leakage.
14. The project currently contains a complete Phase 0 and complete Phase 1 scripting layer, but Phase 2 and Phase 3 experiment runners remain unimplemented placeholders.
