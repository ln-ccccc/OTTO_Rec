import xgboost as xgb
from config import CFG
import pandas as pd
from pathlib import Path

def train_ranker():
    print("Loading Training Data...")
    df = pd.read_pickle(Path(CFG.PROCESSED_DATA_PATH) / "train_features.pkl")
    df = df.sort_values("session").reset_index(drop=True)
    
    # 特征选择：商品热度（pop_count）和是否在历史中（in_history）
    features = ["pop_count", "in_history"]
    target = "label"

    print("Preparing DMatrix...")
    X = df[features]
    y = df[target]

    groups = df.groupby("session").size().to_numpy()
    
    dtrain = xgb.DMatrix(X, y, group=groups)
    
    # XGBoost 参数
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg',
        'learning_rate': 0.1,
        'max_depth': 4,
        'verbosity': 1
    }
    
    print("Training Model...")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 特征重要性
    print("Feature Importance:")
    print(model.get_score(importance_type='gain'))
    
    # 保存模型
    print(f"Saving Model to {CFG.PROCESSED_DATA_PATH}/xgb_ranker.model")
    model.save_model(f"{CFG.PROCESSED_DATA_PATH}/xgb_ranker.model")

if __name__ == "__main__":
    train_ranker()
