import xgboost as xgb
from config import CFG
import pandas as pd
from pathlib import Path

# =============================================================================
# 模块：排序模型训练 (Ranking Layer)
# 作用：训练 XGBoost Ranker，对召回的 50 个候选商品进行精细打分
# 核心思想：Learning to Rank (LTR)
#   - Objective: rank:pairwise (成对排序)
#   - 既然 Click/Cart/Order 的用户意图不同，我们训练三个独立的模型
# =============================================================================

def train_model(df, target_col, model_name):
    """
    通用训练函数
    :param df: 包含特征和标签的 DataFrame
    :param target_col: 要预测的标签列名 (e.g., 'label_click')
    :param model_name: 保存的模型文件名 (e.g., 'xgb_click.model')
    """
    print(f"Training model for {target_col} -> {model_name}...")
    
    # -------------------------------------------------------------------------
    # 1. 特征选择 (Feature Selection)
    # -------------------------------------------------------------------------
    # 加入了 'score' (召回分数)，这是最重要的特征！
    features = ["pop_count", "in_history", "score"]
    
    X = df[features]
    y = df[target_col]
    
    # -------------------------------------------------------------------------
    # 2. 构建 Group 信息 (关键!)
    # -------------------------------------------------------------------------
    # XGBoost 做 Pairwise Ranking 时，必须知道哪些样本属于同一个 Query (Session)
    # 模型只会比较同一个 Group 内部的样本顺序，不会跨 Group 比较
    groups = df.groupby("session").size().to_numpy()
    
    # DMatrix 是 XGBoost 的专用数据格式，支持 group 参数
    dtrain = xgb.DMatrix(X, y, group=groups)
    
    # -------------------------------------------------------------------------
    # 3. 设置参数 (Hyperparameters)
    # -------------------------------------------------------------------------
    params = {
        # 核心目标：成对排序 (让正样本排在负样本前面)
        # 相比 Pointwise (二分类)，它直接优化 NDCG 指标，更适合推荐
        'objective': 'rank:pairwise',
        
        # 评估指标
        'eval_metric': 'ndcg',
        
        # 学习率：越小越稳，但训练越慢
        'learning_rate': 0.1,
        
        # 树深：推荐系统特征通常较稀疏，不需要太深，4-6 足够
        'max_depth': 4,
        
        # 打印日志详细程度
        'verbosity': 1
    }
    
    # -------------------------------------------------------------------------
    # 4. 训练与保存
    # -------------------------------------------------------------------------
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 打印特征重要性 (Feature Importance)
    # Gain: 该特征在树分裂中带来的平均增益 (贡献度)
    print(f"Feature Importance ({target_col}):")
    print(model.get_score(importance_type='gain'))
    
    out_path = Path(CFG.PROCESSED_DATA_PATH) / model_name
    print(f"Saving Model to {out_path}")
    model.save_model(out_path)

def train_all():
    print("Loading Training Data...")
    # 读取在 04_features.py 中生成好的特征集
    df = pd.read_pickle(Path(CFG.PROCESSED_DATA_PATH) / "train_features.pkl")
    
    # 必须按 session 排序，否则 group 信息会乱
    df = df.sort_values("session").reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # 策略调整：训练单模型 (Unified Model)
    # -------------------------------------------------------------------------
    # 原因：label_cart 和 label_order 太稀疏，单独训练容易过拟合或学不到东西。
    # 方案：构建一个 unified label (只要有交互就算 1)，训练一个通用模型。
    # 加上 'score' 特征后，这个模型会非常强大。
    
    print("Creating Unified Label...")
    # 逻辑或：只要发生了点击、加购、购买中的任意一种，都认为是感兴趣
    df["label_action"] = (df["label_click"] + df["label_cart"] + df["label_order"]).clip(upper=1)
    
    # 训练通用模型
    train_model(df, "label_action", "xgb_action.model")

if __name__ == "__main__":
    train_all()
