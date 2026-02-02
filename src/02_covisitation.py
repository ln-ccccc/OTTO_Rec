from config import CFG
import pickle
import gc
from split_utils import load_valid_cutoff
import json
from collections import defaultdict, Counter
from pathlib import Path

""" 构建共现矩阵 """



def build_click_click(train_path, valid_cutoff, line_limit):
    """构建 Click-Click 矩阵 """
    print("Building Click-Click Matrix...")
    neighbors = defaultdict(Counter)
    n_lines = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if line_limit is not None and n_lines > line_limit: break
            
            obj = json.loads(line)
            # 只取 cutoff 之前的事件
            ev = [e for e in obj["events"] if e["ts"] < valid_cutoff]
            if not ev: continue
            
            # 按时间排序，取最近 30 条
            ev.sort(key=lambda x: x["ts"])
            ev = ev[-30:]
            
            # 提取 aid 序列
            aids = [int(e["aid"]) for e in ev]
            if len(aids) < 2: continue
            
            # 相邻共现：A -> B
            for a, b in zip(aids, aids[1:]):
                if a == b: continue
                neighbors[a][b] += 1
                neighbors[b][a] += 1
                
    return neighbors

def build_cart_order(train_path, valid_cutoff, line_limit):
    """
    [任务] 构建 Cart-Order 矩阵
    逻辑：
    1. 筛选 type='carts' 的 aid 集合 -> carts
    2. 筛选 type='orders' 的 aid 集合 -> orders
    3. 双重循环：for c in carts: for o in orders: weight += 1
    """
    print("Building Cart-Order Matrix...")
    neighbors = defaultdict(Counter)
    n_lines = 0
    
    # 类型映射参考 config.py: {'clicks': 0, 'carts': 1, 'orders': 2}
    # 但 jsonl 里 type 是字符串 "carts", "orders"
    
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if line_limit is not None and n_lines > line_limit: break
            
            obj = json.loads(line)
            ev = [e for e in obj["events"] if e["ts"] < valid_cutoff]
            if not ev: continue
            
            # --- 请在此处补充代码 ---
            # 1. 找到该 session 内所有加购过的商品 ID (去重)
            cart_aids = set(e['aid'] for e in ev if e['type'] == 'carts')
            
            # 2. 找到该 session 内所有购买过的商品 ID (去重)
            order_aids = set(e['aid'] for e in ev if e['type'] == 'orders')
            
            # 3. 记录共现 (Cart -> Order)
            for i in cart_aids:
                for j in order_aids:
                    if i == j: continue
                    neighbors[i][j] += 1
                    neighbors[j][i] += 1 # 无向图，互为关联
            
            
            
    return neighbors

def build_buy_buy(train_path, valid_cutoff, line_limit):
    """
    [任务] 构建 Buy-Buy 矩阵
    逻辑：
    1. 筛选 type='orders' 的 aid 集合 -> orders
    2. 双重循环：for a in orders: for b in orders: weight += 1
    """
    print("Building Buy-Buy Matrix...")
    neighbors = defaultdict(Counter)
    n_lines = 0
    
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if line_limit is not None and n_lines > line_limit: break
            
            obj = json.loads(line)
            ev = [e for e in obj["events"] if e["ts"] < valid_cutoff]
            if not ev: continue
            
            # --- 请在此处补充代码 ---
            # 1. 找到该 session 内所有购买过的商品 ID (去重)
            order_aids = set(e['aid'] for e in ev if e['type'] == 'orders')
            
            # 2. 记录共现 (Order <-> Order)
            # 如果 order_aids 长度 < 2，跳过
            if len(order_aids) < 2: continue

            for i in order_aids:
                for j in order_aids:
                    if i == j: continue
                    neighbors[i][j] += 1
                    # neighbors[j][i] += 1 # 不需要写这一行，因为双重循环会覆盖 (j, i)
           
            
    return neighbors

def save_matrix(neighbors, name):
    print(f"Saving {name}...")
    # 只保留 Top N (从 config 读取)
    matrix = {aid: [b for b, _ in cnt.most_common(CFG.TOP_N)] for aid, cnt in neighbors.items()}
    with open(f"{CFG.PROCESSED_DATA_PATH}/covisitation_{name}.pkl", "wb") as f:
        pickle.dump(matrix, f)
    del neighbors, matrix
    gc.collect()

def generate_covisitation():
    print("Loading data for Co-visitation...")
    valid_cutoff = load_valid_cutoff(CFG.PROCESSED_DATA_PATH)
    train_path = Path(CFG.RAW_DATA_PATH) / CFG.TRAIN_FILE
    print(f"Reading from {train_path}...")
    
    # 1. Click-Click
    neighbors_cc = build_click_click(train_path, valid_cutoff, None)
    save_matrix(neighbors_cc, "click_click")
    
    # 2. Cart-Order
    neighbors_co = build_cart_order(train_path, valid_cutoff, None)
    save_matrix(neighbors_co, "cart_order")
    
    # 3. Buy-Buy
    neighbors_bb = build_buy_buy(train_path, valid_cutoff, None)
    save_matrix(neighbors_bb, "buy_buy")

if __name__ == "__main__":
    generate_covisitation()
