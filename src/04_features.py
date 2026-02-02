from config import CFG
from item_stats import load_or_build_item_pop
import pandas as pd
from pathlib import Path

def generate_features():
    print("Loading Candidates...")
    # 加载训练集候选商品
    df_candidates = pd.read_pickle(Path(CFG.PROCESSED_DATA_PATH) / "candidates_train.pkl")
    
    # --- 特征 1：商品热度（全局计数） ---
    print("Calculating Item Popularity...")
    # 加载或计算商品全局点击次数
    pop = load_or_build_item_pop(CFG.RAW_DATA_PATH, CFG.PROCESSED_DATA_PATH)
    # 映射候选商品的点击次数
    df_candidates["pop_count"] = df_candidates["aid"].map(lambda x: pop.get(int(x), 0)).astype("int64")
    # 合并候选商品和特征
    df_features = df_candidates
    
    # 保存训练特征
    print("Saving Features...")
    out_path = Path(CFG.PROCESSED_DATA_PATH) / "train_features.pkl"
    df_features.to_pickle(out_path)
    print(f"Feature set shape: {df_features.shape}")
    print(df_features.head())
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_features()
