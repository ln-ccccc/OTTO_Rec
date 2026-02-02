#!/bin/bash
set -e

# 激活环境 (仅在交互式 shell 中有效，脚本中通常假设环境已激活或使用全路径)
# 如果脚本在 conda 环境外运行，请取消注释下一行并修改路径
# source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate otto

echo "=================================================="
echo "Starting OTTO Recommendation Pipeline"
echo "=================================================="

# 0. 检查依赖
python -c "import pandas, numpy, xgboost, sklearn, tqdm; print('Dependencies check passed')"

# 1. 预处理 (计算时间切分)
echo "--------------------------------------------------"
echo "[Step 1/6] Preprocessing (Time Cutoff)..."
python src/01_preprocess.py

# 2. 生成共现矩阵 (Recall 核心)
echo "--------------------------------------------------"
echo "[Step 2/6] Building Co-visitation Matrices..."
python src/02_covisitation.py

# 3. 生成候选集 (Candidates)
echo "--------------------------------------------------"
echo "[Step 3/6] Generating Candidates..."
python src/03_candidates.py

# 4. 特征工程 (Features)
echo "--------------------------------------------------"
echo "[Step 4/6] Feature Engineering..."
python src/04_features.py

# 5. 训练模型 (Ranking)
echo "--------------------------------------------------"
echo "[Step 5/6] Training Ranker Models (XGBoost)..."
python src/05_train_ranker.py

# 6. 推理 (Inference)
echo "--------------------------------------------------"
echo "[Step 6/6] Inference & Submission..."
python src/06_inference.py

echo "=================================================="
echo "Pipeline Completed Successfully!"
echo "Submission saved to: submission.csv"
echo "=================================================="
