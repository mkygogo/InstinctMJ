#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[INFO] Root directory: ${ROOT_DIR}"
echo "[INFO] Creating virtual environment .venv"
python -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"

python -m pip install -U pip

echo "[INFO] Installing mujoco-warp (required by mjlab)"
python -m pip install "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@b75f85b4dbc60f2d6e9277d345953aa1f1dd1611"

echo "[INFO] Installing mjlab, instinct_rl and instinct_mjlab in editable mode"
python -m pip install -e "${ROOT_DIR}/../mjlab"
python -m pip install -e "${ROOT_DIR}/../instinct_rl"
python -m pip install regex
python -m pip install -e "${ROOT_DIR}"

echo "[INFO] Done."
echo "[INFO] Activate env with: source ${ROOT_DIR}/.venv/bin/activate"
echo "[INFO] Then run: instinct-list-envs"
