from config import CFG
import pickle
from collections import Counter
from split_utils import load_valid_cutoff
import json
from pathlib import Path
import pandas as pd
import os

# =============================================================================
# 模块：候选集生成 (Recall Layer)
# 作用：从海量商品(1.8M+)中筛选出 Top-N(50) 候选，并为训练集打上真实标签(Label)
# 核心思想：多路召回 (Multi-channel Recall)
#   1. 历史回溯 (History): 用户看过什么，大概率还会看/买什么。
#   2. 共现矩阵 (Co-visitation): 看了A的人，通常也会看B (Item-to-Item)。
# =============================================================================

def load_matrix(name):
    """安全加载共现矩阵，如果文件不存在返回空字典"""
    path = f"{CFG.PROCESSED_DATA_PATH}/covisitation_{name}.pkl"
    if not os.path.exists(path):
        print(f"Warning: Matrix {name} not found, returning empty dict.")
        return {}
    print(f"Loading {name}...")
    with open(path, "rb") as f:
        return pickle.load(f)

def generate_candidates():
    # -------------------------------------------------------------------------
    # 1. 准备工作：加载资源
    # -------------------------------------------------------------------------
    # 加载三个不同维度的共现矩阵 (Recall Sources)
    cov_cc = load_matrix("click_click") # 点击->点击 (浏览兴趣漂移)
    cov_co = load_matrix("cart_order")  # 加购->购买 (强转化关联)
    cov_bb = load_matrix("buy_buy")     # 购买->购买 (搭配购买)
    
    # 获取验证集的时间切分点 (Cutoff Timestamp)
    # 我们用这个时间点把 train.jsonl 切成两半：
    # - Past (History): 用于生成候选 (输入)
    # - Future (Truth): 用于验证预测是否正确 (标签)
    valid_cutoff = load_valid_cutoff(CFG.PROCESSED_DATA_PATH)

    print("Building labels (future events)...")
    train_path = Path(CFG.RAW_DATA_PATH) / CFG.TRAIN_FILE
    print(f"Reading from {train_path}...")

    # -------------------------------------------------------------------------
    # 2. 构建 Ground Truth (真实标签)
    # 目标：知道每个 Session 在 Cutoff 之后真正点击/加购/购买了哪些商品
    # 结构：truth[session_id] = {'clicks': {aid1, aid2}, 'carts': {...}, 'orders': {...}}
    # -------------------------------------------------------------------------
    truth = {}
    n_lines = 0
    # 遍历训练集，构建每个 Session 的真实标签
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            s = int(obj["session"])
            
            # 初始化 Session 的真实标签容器
            # 每个 Session 有三种行为：点击、加购、购买
            if s not in truth:
                truth[s] = {'clicks': set(), 'carts': set(), 'orders': set()}
            # 遍历 Session 中的每个事件
            for e in obj["events"]:
                # 关键：只记录 Cutoff 之后的行为作为“未来真相”
                if e["ts"] >= valid_cutoff:
                    t_type = e['type'] # clicks/carts/orders
                    truth[s][t_type].add(int(e["aid"]))

    # 过滤：只保留那些在未来有行为的 Session 用于训练
    # 如果一个用户在 Cutoff 后就消失了，那他无法提供监督信号，训练它没意义
    valid_sessions = {s for s, t in truth.items() if (t['clicks'] or t['carts'] or t['orders'])}

    # -------------------------------------------------------------------------
    # 3. 构建 History (用户历史)
    # 目标：获取用户在 Cutoff 之前的行为序列，作为召回的“种子”
    # -------------------------------------------------------------------------
    print("Loading history...")
    history = {}
    n_lines = 0
    # 遍历训练集，构建每个 Session 的历史行为序列
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            s = int(obj["session"])
            if s not in valid_sessions: continue
            
            # 关键：只取 Cutoff 之前的行为作为“历史输入”
            ev = [e for e in obj["events"] if e["ts"] < valid_cutoff]
            if not ev: continue
            # 按时间排序：最近的行为权重最高
            ev.sort(key=lambda x: x["ts"])
            history[s] = [int(e["aid"]) for e in ev]

    # -------------------------------------------------------------------------
    # 4. 多路召回主循环 (Multi-Channel Recall)
    # 核心逻辑：对每个 Session，综合多种策略生成 Top-N 候选
    # -------------------------------------------------------------------------
    print("Generating candidates with Multi-Recall...")
    
    # 优化：使用列式存储代替行式存储 (List of Dicts -> Dict of Lists)
    # 这种方式比 list[dict] 节省大量内存
    columns = {
        "session": [],
        "aid": [],
        "score": [], # 新增：召回分数 (Rule-based Score)
        "in_history": [],
        "label_click": [],
        "label_cart": [],
        "label_order": []
    }
    
    # 用于统计召回率 (Metrics)
    total_truth = {'clicks': 0, 'carts': 0, 'orders': 0}
    hit_truth = {'clicks': 0, 'carts': 0, 'orders': 0}
    
    # 分块处理：避免一次性在大内存中持有所有 list
    dfs = [] 
    CHUNK_SIZE = 20000
    count = 0
    
    # 遍历所有需要预测的历史 Session
    # 使用 list(history.keys()) 允许我们在循环中安全地访问数据
    session_ids = list(history.keys())
    
    for s in session_ids:
        events = history[s]
        count += 1
        
        # 种子去重：虽然用户可能多次点击同一个商品，但作为种子只算一个
        unique_events = list(dict.fromkeys(events))
        history_set = set(unique_events) # 用于快速查找是否看过
        candidates = Counter()
        
        # --- 策略 A: 历史回溯 (History) ---
        # 逻辑：用户大概率会去买他刚才看过的东西
        # 权重：给一个很高的保底分 (10分)，确保历史商品优先排在前面
        for aid in unique_events[::-1]: # 倒序，最近的优先
            candidates[aid] += 10
            
        # --- 策略 B: 矩阵扩散 (Co-visitation) ---
        # 逻辑：拿着用户看过的商品(种子)，去矩阵里查“邻居”
        for aid in unique_events[::-1]:
            # 路1: Click-Click (权重=1，最弱)
            for neigh in cov_cc.get(aid, []):
                if neigh not in history_set: # 排除掉已经看过的，因为已经在策略A里加过分了
                    candidates[neigh] += 1
            
            # 路2: Cart-Order (权重=3，较强)
            for neigh in cov_co.get(aid, []):
                if neigh not in history_set:
                    candidates[neigh] += 3
            
            # 路3: Buy-Buy (权重=5，最强)
            # 买了A的人通常必买B，这个信号非常强烈
            for neigh in cov_bb.get(aid, []):
                if neigh not in history_set:
                    candidates[neigh] += 5

        # --- 截断 (Truncation) ---
        # 只保留分数最高的 Top-N (50) 个商品
        top_candidates = [aid for aid, _ in candidates.most_common(CFG.TOP_N)]
        
        # --- 标签构造 (Label Construction) ---
        # 检查这 50 个候选商品，在未来(Truth)是否真的发生了行为
        t_clicks = truth[s]['clicks']
        t_carts = truth[s]['carts']
        t_orders = truth[s]['orders']
        
        # 优化：处理完即删除，释放内存
        del truth[s]
        
        # 实时统计召回率 (Recall Calculation)
        # Recall = 命中的正样本数 / 总正样本数
        # 例如：用户点击了 A B C，我们召回了 A B D，那么 Recall = 2/3
        if t_clicks:
            total_truth['clicks'] += 1
            if any(aid in t_clicks for aid in top_candidates): hit_truth['clicks'] += 1
        if t_carts:
            total_truth['carts'] += 1
            if any(aid in t_carts for aid in top_candidates): hit_truth['carts'] += 1
        if t_orders:
            total_truth['orders'] += 1
            if any(aid in t_orders for aid in top_candidates): hit_truth['orders'] += 1
        
        # 写入训练样本 (列式追加)
        for aid in top_candidates:
            # 每个商品的特征：
            # 1. session_id
            # 2. aid (候选商品)
            # 3. 是否在历史序列里 (1/0)
            # 4. 未来是否点击 (1/0)
            # 5. 未来是否加入购物车 (1/0)
            # 6. 未来是否下单 (1/0)
            columns["session"].append(s)
            columns["aid"].append(aid)
            columns["score"].append(candidates[aid]) # 保存分数
            columns["in_history"].append(1 if aid in history_set else 0)
            columns["label_click"].append(1 if aid in t_clicks else 0)
            columns["label_cart"].append(1 if aid in t_carts else 0)
            columns["label_order"].append(1 if aid in t_orders else 0)

        # 分块保存 (Chunking)
        if count % CHUNK_SIZE == 0:
            print(f"Processed {count} sessions...")
            chunk_df = pd.DataFrame(columns)
            # 转换数据类型以节省内存
            chunk_df['session'] = chunk_df['session'].astype('int32')
            chunk_df['aid'] = chunk_df['aid'].astype('int32')
            chunk_df['score'] = chunk_df['score'].astype('float32') # 分数
            chunk_df['in_history'] = chunk_df['in_history'].astype('int8')
            chunk_df['label_click'] = chunk_df['label_click'].astype('int8')
            chunk_df['label_cart'] = chunk_df['label_cart'].astype('int8')
            chunk_df['label_order'] = chunk_df['label_order'].astype('int8')
            dfs.append(chunk_df)
            
            # 清空当前 buffer
            for k in columns:
                columns[k] = []

    # 处理剩余数据
    if columns["session"]:
        chunk_df = pd.DataFrame(columns)
        chunk_df['session'] = chunk_df['session'].astype('int32')
        chunk_df['aid'] = chunk_df['aid'].astype('int32')
        chunk_df['score'] = chunk_df['score'].astype('float32')
        chunk_df['in_history'] = chunk_df['in_history'].astype('int8')
        chunk_df['label_click'] = chunk_df['label_click'].astype('int8')
        chunk_df['label_cart'] = chunk_df['label_cart'].astype('int8')
        chunk_df['label_order'] = chunk_df['label_order'].astype('int8')
        dfs.append(chunk_df)

    # -------------------------------------------------------------------------
    # 5. 保存结果
    # -------------------------------------------------------------------------
    if dfs:
        df_candidates = pd.concat(dfs, ignore_index=True)
    else:
        df_candidates = pd.DataFrame(columns) # Empty fallback

    print(f"Candidates shape: {df_candidates.shape}")
    # 打印正样本数量，如果太少说明召回策略有问题
    print(f"Positive Clicks: {df_candidates['label_click'].sum()}")
    print(f"Positive Carts:  {df_candidates['label_cart'].sum()}")
    print(f"Positive Orders: {df_candidates['label_order'].sum()}")
    
    # 打印最终召回率
    for t in ['clicks', 'carts', 'orders']:
        if total_truth[t] > 0:
            print(f"Recall@{CFG.TOP_N} ({t}): {hit_truth[t]/total_truth[t]:.4f} ({hit_truth[t]}/{total_truth[t]})")
    
    out_path = Path(CFG.PROCESSED_DATA_PATH) / "candidates_train.pkl"
    df_candidates.to_pickle(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_candidates()
    