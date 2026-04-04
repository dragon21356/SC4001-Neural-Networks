#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="kan-vs-mlp"

mkdir -p \
  "${ROOT_DIR}/src" \
  "${ROOT_DIR}/experiments/configs" \
  "${ROOT_DIR}/notebooks" \
  "${ROOT_DIR}/results/exp1" \
  "${ROOT_DIR}/results/exp2" \
  "${ROOT_DIR}/results/exp3" \
  "${ROOT_DIR}/report/figures"

touch \
  "${ROOT_DIR}/src/__init__.py" \
  "${ROOT_DIR}/src/kan_layer.py" \
  "${ROOT_DIR}/src/bspline_activation.py" \
  "${ROOT_DIR}/src/models.py" \
  "${ROOT_DIR}/src/data_utils.py" \
  "${ROOT_DIR}/src/train.py" \
  "${ROOT_DIR}/src/evaluate.py" \
  "${ROOT_DIR}/src/spline_vis.py" \
  "${ROOT_DIR}/src/utils.py" \
  "${ROOT_DIR}/experiments/exp1_regression.py" \
  "${ROOT_DIR}/experiments/exp2_cifar10.py" \
  "${ROOT_DIR}/experiments/exp3_interpretability.py" \
  "${ROOT_DIR}/experiments/configs/exp1_config.yaml" \
  "${ROOT_DIR}/experiments/configs/exp2_config.yaml" \
  "${ROOT_DIR}/notebooks/Exp1_Regression.ipynb" \
  "${ROOT_DIR}/notebooks/Exp2_CIFAR10.ipynb" \
  "${ROOT_DIR}/notebooks/Exp3_Interpretability.ipynb" \
  "${ROOT_DIR}/notebooks/Results_Visualization.ipynb" \
  "${ROOT_DIR}/requirements.txt" \
  "${ROOT_DIR}/README.md"

echo "Project scaffold created at ${ROOT_DIR}/"
