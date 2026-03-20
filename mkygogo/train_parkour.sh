#!/bin/bash

# 遇到任何报错立刻终止脚本
set -e

echo ">>> [1/4] 正在解析动态路径..."
# 获取当前脚本所在的绝对路径 (InstinctMJ/mkygogo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 向上推导一级，获取项目根目录 (InstinctMJ)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 向上推导并定位到同级的数据集目录 (InstinctMJ/../data/parkour_motion_reference)
DATA_DIR="$(cd "$PROJECT_DIR/../data/parkour_motion_reference" && pwd)"

CONFIG_FILE="$PROJECT_DIR/src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py"

echo "项目根目录已解析为: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo ">>> [2/4] 激活运行环境..."
# 初始化并激活 Conda 基础环境
eval "$(conda shell.bash hook)"
conda activate instinct_env

# 激活 uv 创建的本地虚拟环境
source .venv/bin/activate

echo ">>> [3/4] 自动更新数据集配置..."
# 使用 sed 命令，自动将配置文件中的数据集变量替换为解析出的绝对路径
sed -i "s|^_PARKOUR_DATASET_DIR = .*|_PARKOUR_DATASET_DIR = \"$DATA_DIR\"|g" "$CONFIG_FILE"
echo "已成功将数据集路径指向: $DATA_DIR"

echo ">>> [4/4] 启动强化学习训练任务..."
# 使用 1024 个并行环境以适配 12GB 显存
NUM_ENVS="${NUM_ENVS:-1024}"
instinct-train Instinct-Parkour-Target-Amp-G1-v0 --num-envs "$NUM_ENVS"
echo ">>> 训练进程已结束或被手动终止。"