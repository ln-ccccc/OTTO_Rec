import xgboost as xgb
from config import CFG
import pickle
from collections import Counter
from item_stats import load_or_build_item_pop
import heapq
import json
from pathlib import Path
import numpy as np
import os

# =============================================================================
# 模块：推理与预测 (Inference Layer)
# 作用：对测试集(Test Set)进行全流程预测，生成提交文件(Submission)
# 流程：
#   1. 读取 Test JSONL (逐行流式读取，防OOM)
#   2. 生成候选 (Recall): 逻辑与训练时一致
#   3. 特征计算 (Feature Engineering)
#   4. 模型打分 (Ranking): 分别调用 Click/Cart/Order 三个模型
#   5. 格式化输出 (Formatting)
# =============================================================================

# Helper to load matrix safely
def load_matrix(name):
    path = f"{CFG.PROCESSED_DATA_PATH}/covisitation_{name}.pkl"
    if not os.path.exists(path): return {}
    with open(path, "rb") as f: return pickle.load(f)

def inference():
    print("Loading Test Data...")
    
    # -------------------------------------------------------------------------
    # 1. 预加载资源 (Matrices & Models)
    # -------------------------------------------------------------------------
    print("Loading Co-visitation Matrices...")
    cov_cc = load_matrix("click_click")
    cov_co = load_matrix("cart_order")
    cov_bb = load_matrix("buy_buy")
        
    print("Loading Item Popularity from Train...")
    pop = load_or_build_item_pop(CFG.RAW_DATA_PATH, CFG.PROCESSED_DATA_PATH)
    
    # 冷启动兜底策略 (Fallback Strategy)
    # 如果一个用户没有任何历史行为，或者候选集不足 20 个，
    # 我们就用“全局最热的 50 个商品”来填充
    print("Calculating Global Top 50 Fallback...")
    top_50_list = [aid for aid, _ in heapq.nlargest(50, pop.items(), key=lambda x: x[1])]
    top_50_str = " ".join(str(x) for x in top_50_list)
    print(f"Top 50 global: {top_50_str[:50]}...")

    print("Loading Models...")
    # 策略调整：加载单模型 (Unified Model)
    model = xgb.Booster()
    model.load_model(f"{CFG.PROCESSED_DATA_PATH}/xgb_action.model")
    
    test_path = Path(CFG.RAW_DATA_PATH) / CFG.TEST_FILE
    print(f"Reading from {test_path}...")

    # -------------------------------------------------------------------------
    # 2. 批处理循环 (Batch Processing)
    # 为了处理 1.6M 个 Session 而不爆内存，我们采用 Batch 模式：
    # 积攒 20,000 个 Session -> 统一生成特征 -> 统一打分 -> 统一写入 -> 清空内存
    # -------------------------------------------------------------------------
    batch_sessions = []
    batch_aids = []      # 候选商品ID (用于排序后输出)
    batch_scores = []    # 新增特征: 召回分数
    batch_in_hist = []   # 特征: 是否在历史中
    batch_pop = []       # 特征: 热度
    batch_offsets = [0]  # 记录每个 Session 在 batch_aids 中的起始位置 (CSR 格式)
    
    # 统计计数器
    stats = {'sessions_processed': 0}

    def flush(out_f):
        """将当前 Batch 的数据送入模型预测并写入文件"""
        if not batch_sessions: return
        
        # 异常防御：如果 batch_aids 为空（说明所有 Session 都没有候选且兜底失败）
        # 这种情况理论上不该发生，但为了防止漏行，必须处理
        if not batch_aids:
            print(f"WARNING: Batch of {len(batch_sessions)} sessions has NO candidates! Using raw fallback.")
            for s in batch_sessions:
                out_f.write(f"{s}_clicks,{top_50_str}\n")
                out_f.write(f"{s}_carts,{top_50_str}\n")
                out_f.write(f"{s}_orders,{top_50_str}\n")
                stats['sessions_processed'] += 1
            
            # 清理并返回
            batch_sessions.clear()
            batch_offsets[:] = [0]
            return
        
        # 2.1 构建特征矩阵 (Feature Matrix)
        # 特征顺序必须和 05_train_ranker.py 一致: ["pop_count", "in_history", "score"]
        X = np.column_stack([
            np.array(batch_pop, dtype=np.float32), 
            np.array(batch_in_hist, dtype=np.float32),
            np.array(batch_scores, dtype=np.float32)
        ])
        dtest = xgb.DMatrix(X, feature_names=["pop_count", "in_history", "score"])
        
        # 2.2 统一预测
        # 使用单模型预测“感兴趣程度”
        scores = model.predict(dtest)
        
        # 2.3 结果组装 (Result Assembly)
        for i, s in enumerate(batch_sessions):
            # 获取当前 Session 在 batch 中的切片范围
            start = batch_offsets[i]
            end = batch_offsets[i + 1]
            aids = batch_aids[start:end]
            
            # 内部函数：根据分数排序取 Top 20
            def get_top_str(sc):
                # sc = scores[start:end] # 这里复用传入的 scores 切片
                if len(sc) == 0: return top_50_str # 防御空切片
                order = np.argsort(-sc) # 降序排列的索引
                top = [str(aids[j]) for j in order[:20]] # 取 Top 20 商品 ID
                return " ".join(top) if top else top_50_str # 兜底

            # 获取当前 session 的分数切片
            current_scores = scores[start:end]
            top_str = get_top_str(current_scores)
            
            # 三个任务使用相同的排序结果 (强基线)
            out_f.write(f"{s}_clicks,{top_str}\n")
            out_f.write(f"{s}_carts,{top_str}\n")
            out_f.write(f"{s}_orders,{top_str}\n")
            
            stats['sessions_processed'] += 1

        # 2.4 清空 Batch
        batch_sessions.clear()
        batch_aids.clear()
        batch_scores.clear()
        batch_in_hist.clear()
        batch_pop.clear()
        batch_offsets[:] = [0]

    print("Writing submission.csv...")
    with open("submission.csv", "w", encoding="utf-8") as out_f:
        out_f.write("session_type,labels\n")

        n_lines = 0
        with test_path.open("r", encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                try:
                    obj = json.loads(line)
                    s = int(obj["session"])
                    ev = obj["events"]
                    # 容错：如果 events 为空
                    if not ev:
                        unique_events = []
                    else:
                        ev.sort(key=lambda x: x["ts"])
                        aids = [int(e["aid"]) for e in ev]
                        unique_events = list(dict.fromkeys(aids))
                    
                    history_set = set(unique_events)
                except Exception as e:
                    print(f"Error parsing line {n_lines}: {e}")
                    continue

                # --- 候选生成 (Recall) ---
                # 逻辑与 03_candidates.py 完全一致
                candidates = Counter()
                # 1. 历史
                for aid in unique_events[::-1]:
                    candidates[aid] += 20
                # 2. 矩阵
                for aid in unique_events[::-1]:
                    for neigh in cov_cc.get(aid, []):
                        if neigh not in history_set: candidates[neigh] += 1
                    for neigh in cov_co.get(aid, []):
                        if neigh not in history_set: candidates[neigh] += 3
                    for neigh in cov_bb.get(aid, []):
                        if neigh not in history_set: candidates[neigh] += 5

                # 截断 Top 50
                # 注意：这里需要同时获取 aid 和 score
                top_candidates_with_score = candidates.most_common(CFG.TOP_N)
                
                # 如果完全没召回（极其罕见），用 Top 50 填充
                if not top_candidates_with_score:
                    # Top 50 兜底时，给一个默认分数 1.0 (虽然不太重要，但要占位)
                    top_candidates_with_score = [(aid, 1.0) for aid in top_50_list[:CFG.TOP_N]]

                # --- 特征收集 (Feature Collection) ---
                # 将当前 Session 的数据加入 Batch
                batch_sessions.append(s)
                for aid, score in top_candidates_with_score:
                    batch_aids.append(aid)
                    batch_scores.append(score) # 保存分数
                    # 实时计算特征
                    batch_in_hist.append(1 if aid in history_set else 0)
                    batch_pop.append(pop.get(aid, 0))
                
                # 更新 Offset
                batch_offsets.append(len(batch_aids))

                # 积攒够了就 Flush
                if len(batch_sessions) >= 20_000:
                    flush(out_f)
                    print(f"Processed {n_lines} lines...")

        # 处理剩余的 Session
        flush(out_f)

    print(f"Submission saved to submission.csv. Total sessions: {stats['sessions_processed']}")

if __name__ == "__main__":
    inference()
