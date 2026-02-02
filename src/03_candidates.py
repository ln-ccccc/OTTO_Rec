from config import CFG
import pickle
from collections import Counter
from split_utils import load_valid_cutoff
import json
from pathlib import Path
import pandas as pd
import os

def load_matrix(name):
    path = f"{CFG.PROCESSED_DATA_PATH}/covisitation_{name}.pkl"
    if not os.path.exists(path):
        print(f"Warning: Matrix {name} not found, returning empty dict.")
        return {}
    print(f"Loading {name}...")
    with open(path, "rb") as f:
        return pickle.load(f)

def generate_candidates():
    # 1. 加载三个矩阵
    cov_cc = load_matrix("click_click")
    cov_co = load_matrix("cart_order")
    cov_bb = load_matrix("buy_buy")
    
    valid_cutoff = load_valid_cutoff(CFG.PROCESSED_DATA_PATH)

    print("Building labels (future events)...")
    train_path = Path(CFG.RAW_DATA_PATH) / CFG.TRAIN_FILE
    print(f"Reading from {train_path}...")

    # 2. 构建 Future Events (Truth)
    # 格式：{session: set(aids)}
    truth = {}
    n_lines = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            s = int(obj["session"])
            tset = None
            for e in obj["events"]:
                if e["ts"] >= valid_cutoff:
                    if tset is None:
                        tset = truth.get(s)
                        if tset is None:
                            tset = set()
                            truth[s] = tset
                    tset.add(int(e["aid"]))

    valid_sessions = set(truth.keys())

    print("Loading history...")
    # 3. 构建历史事件序列
    # 格式：{session: [aid1, aid2, ...]}
    history = {}
    n_lines = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            s = int(obj["session"])
            if s not in valid_sessions: continue
            ev = [e for e in obj["events"] if e["ts"] < valid_cutoff]
            if not ev: continue
            ev.sort(key=lambda x: x["ts"])
            history[s] = [int(e["aid"]) for e in ev]

    print("Generating candidates with Multi-Recall...")
    candidates_rows = []
    
    # 指标统计
    total_sessions_with_truth = 0
    hit_sessions = 0
    
    # 遍历每个 Session 的历史
    for s, events in history.items():
        unique_events = list(dict.fromkeys(events))
        history_set = set(unique_events)
        candidates = Counter()
        
        # 1. 历史回溯 (Recall Strategy: History)
        # 权重给最高，因为复购概率最大
        for aid in unique_events[::-1]:
            candidates[aid] += 20  # 给个保底分
            
        # 2. 多路矩阵召回 (Recall Strategy: Co-visitation)
        # 遍历历史中的每一个商品，去查它的邻居
        for aid in unique_events[::-1]: # 倒序，优先查最近看过的商品
            
            # --- [任务] 请在此处补充代码 ---
            # 逻辑：
            # 1. 查 Click-Click 矩阵 (权重=1)
            # 2. 查 Cart-Order 矩阵 (权重=3)
            # 3. 查 Buy-Buy 矩阵 (权重=3)
            for neigh in cov_cc.get(aid, []):
                if neigh not in history_set:
                    candidates[neigh] += 1
            
            # 2. 查 Cart-Order 矩阵 (权重=3)
            for neigh in cov_co.get(aid, []):
                if neigh not in history_set:
                    candidates[neigh] += 3
            
            # 3. 查 Buy-Buy 矩阵 (权重=3)
            for neigh in cov_bb.get(aid, []):
                if neigh not in history_set:
                    candidates[neigh] += 5
            
            # 示例伪代码：
            # for neigh in cov_cc.get(aid, []):
            #     if neigh not in history_set:
            #         candidates[neigh] += 1
            
            # 请补全 Cart-Order 和 Buy-Buy 的逻辑
            

        # 取 Top N
        top_candidates = [aid for aid, _ in candidates.most_common(CFG.TOP_N)]
        tset = truth.get(s, set())
        
        # 统计召回率
        if len(tset) > 0:
            total_sessions_with_truth += 1
            # 只要有一个真实商品被召回了，就算 Hit
            # (严格来说 Recall 是 命中数/总真实数，HitRate 是 是否命中)
            # 这里简单统计 HitRate@N
            hits = [aid for aid in top_candidates if aid in tset]
            if len(hits) > 0:
                hit_sessions += 1
        
        for aid in top_candidates:
            candidates_rows.append(
                {"session": s, "aid": aid, "in_history": 1 if aid in history_set else 0, "label": 1 if aid in tset else 0}
            )

    df_candidates = pd.DataFrame(candidates_rows)
    print(f"Candidates shape: {df_candidates.shape}")
    print(f"Positive labels: {int(df_candidates['label'].sum())}")
    
    if total_sessions_with_truth > 0:
        recall = hit_sessions / total_sessions_with_truth
        print(f"Recall@{CFG.TOP_N} (HitRate): {recall:.4f} ({hit_sessions}/{total_sessions_with_truth})")
    
    out_path = Path(CFG.PROCESSED_DATA_PATH) / "candidates_train.pkl"
    df_candidates.to_pickle(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_candidates()
