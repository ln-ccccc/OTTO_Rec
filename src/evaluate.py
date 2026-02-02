import xgboost as xgb
from config import CFG
import pandas as pd
from pathlib import Path
import numpy as np

def calculate_mrr(df):
    """
    计算 MRR@20 和 Recall@20
    df 必须包含: session, aid, label, score
    """
    print("Calculating metrics...")
    
    # 1. 按 session 分组，按 score 降序
    # 这里的技巧是：直接利用 pandas 的 groupby apply
    # 但 apply 比较慢，我们可以先 sort
    
    df = df.sort_values(["session", "score"], ascending=[True, False])
    
    # 2. 获取每个 session 的真实正样本 (Ground Truth)
    # 注意：我们的 train_features 里只有 label=1 的才算正样本
    # 但 train_features 已经是候选集了，如果真实正样本没有被召回，这里是不知道的。
    # **这是一个陷阱**：直接用 train_features 算出来的 Recall 是 "Ranking Recall"，不是 "Global Recall"。
    # Global Recall 必须基于原始的 Truth Set (从 jsonl 读出来的)。
    # 但为了简单评估 Ranking 模型的好坏，我们先只算 "在召回集里的排序能力"。
    
    mrr_sum = 0
    hit_sum = 0
    n_sessions = 0
    
    # 这种遍历比较慢，但逻辑最清晰
    # 优化方案：转成 list 或 numpy 处理
    current_session = -1
    ranks = []
    labels = []
    
    # 提取 numpy 数组加速
    session_arr = df["session"].values
    label_arr = df["label"].values
    
    # 找出 session 变化的边界索引
    # np.where(diff)[0] + 1
    change_indices = np.where(session_arr[:-1] != session_arr[1:])[0] + 1
    # 加上开头和结尾
    split_indices = np.concatenate(([0], change_indices, [len(session_arr)]))
    
    print(f"Evaluating {len(split_indices)-1} sessions...")
    
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        
        # 当前 session 的前 20 个候选的 label
        # 因为已经按 score 降序排了，所以直接取前 20
        # 实际取的时候要注意，可能不足 20 个
        sl_labels = label_arr[start:min(end, start + 20)]
        
        # 检查是否有正样本
        if np.sum(sl_labels) > 0:
            hit_sum += 1
            # 计算 MRR: 找到第一个 label=1 的位置 (1-based)
            # np.argmax 返回第一个最大值(1)的索引
            first_hit_idx = np.argmax(sl_labels == 1) + 1
            mrr_sum += (1.0 / first_hit_idx)
            
        n_sessions += 1
        
    print(f"="*30)
    print(f"Ranking Metrics (on Validation Set)")
    print(f"="*30)
    print(f"Sessions evaluated: {n_sessions}")
    print(f"Recall@20 (HitRate): {hit_sum / n_sessions:.4f}")
    print(f"MRR@20:              {mrr_sum / n_sessions:.4f}")
    print(f"="*30)

def evaluate():
    print("Loading Validation Data (with features & labels)...")
    df = pd.read_pickle(Path(CFG.PROCESSED_DATA_PATH) / "train_features.pkl")
    
    print("Loading Model...")
    model = xgb.Booster()
    model.load_model(f"{CFG.PROCESSED_DATA_PATH}/xgb_ranker.model")
    
    features_col = ["pop_count", "in_history"]
    # 确保特征列顺序一致
    # 最好在训练时保存 feature names，这里简化处理
    
    print("Predicting scores...")
    X = df[features_col]
    dtest = xgb.DMatrix(X)
    scores = model.predict(dtest)
    
    df["score"] = scores
    
    calculate_mrr(df)

if __name__ == "__main__":
    evaluate()
