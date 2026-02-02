import json
from config import CFG
from pathlib import Path

def preprocess():
    print("Processing Train Data...")
    
    # 使用 config 中动态配置的文件名
    train_path = Path(CFG.RAW_DATA_PATH) / CFG.TRAIN_FILE
    out_path = Path(CFG.PROCESSED_DATA_PATH) / "valid_cutoff.json"
    print(f"Reading from {train_path}...")
    
    # 只需要扫描一遍拿到 max_ts 即可
    max_ts = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for e in obj["events"]:
                if e["ts"] > max_ts:
                    max_ts = e["ts"]

    valid_cutoff = max_ts - 604_800_000
    out_path.write_text(json.dumps({"max_ts": max_ts, "valid_cutoff": valid_cutoff}), encoding="utf-8")
    print(f"max_ts={max_ts} valid_cutoff={valid_cutoff} saved_to={out_path}")

if __name__ == "__main__":
    preprocess()
