from config import CFG
from item_stats import load_or_build_item_pop
import pandas as pd
from pathlib import Path

def generate_features():
    print("Loading Candidates...")
    df_candidates = pd.read_pickle(Path(CFG.PROCESSED_DATA_PATH) / "candidates_train.pkl")
    
    # --- 特征 1：商品热度（全局计数） ---
    print("Calculating Item Popularity...")
    pop = load_or_build_item_pop(CFG.RAW_DATA_PATH, CFG.PROCESSED_DATA_PATH)
    df_candidates["pop_count"] = df_candidates["aid"].map(lambda x: pop.get(int(x), 0)).astype("int64")
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
