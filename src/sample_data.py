import json
import random
from pathlib import Path
from config import CFG
import os

def sample_debug_data(n_sessions=5000):
    """
    从 train.jsonl 中随机采样 n_sessions 个完整会话，保存为 train_debug.jsonl。
    用于快速验证算法效果（Recall/NDCG），保证 session 完整性。
    """
    print(f"Sampling {n_sessions} sessions for DEBUG mode...")
    
    input_path = Path(CFG.RAW_DATA_PATH) / "train.jsonl"
    output_path = Path(CFG.RAW_DATA_PATH) / "train_debug.jsonl"
    test_output_path = Path(CFG.RAW_DATA_PATH) / "test_debug.jsonl"
    
    # 1. 第一遍扫描：收集所有 session_id
    all_sessions = []
    print("Scanning sessions...")
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # 为了速度，我们可以只扫前 200万行 (足够随机了)
            if i > 2_000_000: 
                break
            # 快速提取 session id (假设 json 结构固定，不做完整 parse 以加速)
            # {"session": 0, ...}
            try:
                # 简单解析，或者完整解析
                obj = json.loads(line)
                all_sessions.append(obj["session"])
            except:
                continue
                
    print(f"Scanned {len(all_sessions)} sessions.")
    
    # 2. 随机抽样
    if len(all_sessions) > n_sessions:
        target_sessions = set(random.sample(all_sessions, n_sessions))
    else:
        target_sessions = set(all_sessions)
        
    print(f"Selected {len(target_sessions)} target sessions.")
    
    # 3. 第二遍扫描：提取目标 session
    print("Extracting data...")
    valid_lines = []
    
    # 重新从头读
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 2_000_000: break
            try:
                obj = json.loads(line)
                if obj["session"] in target_sessions:
                    valid_lines.append(line)
            except:
                continue
                
    # 4. 保存 train_debug.jsonl
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(valid_lines)
        
    print(f"Saved {len(valid_lines)} lines to {output_path}")
    
    # 5. 同时生成配套的 test_debug.jsonl
    # 简单起见，把 train_debug 的后半部分作为 test_debug (模拟)
    # 或者直接复制一份，反正 inference 也是跑通流程
    with test_output_path.open("w", encoding="utf-8") as f:
        f.writelines(valid_lines[:1000]) # 取一部分作为 test
    
    print(f"Saved {1000} lines to {test_output_path}")

if __name__ == "__main__":
    sample_debug_data()
