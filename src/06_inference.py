import xgboost as xgb
from config import CFG
import pickle
from collections import Counter
from item_stats import load_or_build_item_pop
import heapq
import json
from pathlib import Path
import numpy as np

def inference():
    print("Loading Test Data...")
    print("Loading Co-visitation Matrix...")
    with open(f"{CFG.PROCESSED_DATA_PATH}/covisitation_click_click.pkl", "rb") as f:
        covisitation_dict = pickle.load(f)
        
    print("Loading Item Popularity from Train...")
    pop = load_or_build_item_pop(CFG.RAW_DATA_PATH, CFG.PROCESSED_DATA_PATH)
    
    # 兜底：全局 Top50 热门商品（用于冷启动或候选为空）
    print("Calculating Global Top 50 Fallback...")
    top_50_list = [aid for aid, _ in heapq.nlargest(50, pop.items(), key=lambda x: x[1])]
    top_50_str = " ".join(str(x) for x in top_50_list)
    print(f"Top 50 global: {top_50_str[:50]}...")

    print("Loading Model...")
    print("Loading Model...")
    model = xgb.Booster()
    model.load_model(f"{CFG.PROCESSED_DATA_PATH}/xgb_ranker.model")
    
    test_path = Path(CFG.RAW_DATA_PATH) / CFG.TEST_FILE
    print(f"Reading from {test_path}...")

    batch_sessions = []
    batch_aids = []
    batch_in_hist = []
    batch_pop = []
    batch_offsets = [0]

    def flush(out_f):
        if not batch_aids:
            return
        X = np.column_stack([np.array(batch_pop, dtype=np.float32), np.array(batch_in_hist, dtype=np.float32)])
        dtest = xgb.DMatrix(X, feature_names=["pop_count", "in_history"])
        scores = model.predict(dtest)
        for i, s in enumerate(batch_sessions):
            start = batch_offsets[i]
            end = batch_offsets[i + 1]
            aids = batch_aids[start:end]
            sc = scores[start:end]
            order = np.argsort(-sc)
            top = [str(aids[j]) for j in order[:20]]
            label_str = " ".join(top) if top else top_50_str
            out_f.write(f"{s}_clicks,{label_str}\n")
            out_f.write(f"{s}_carts,{label_str}\n")
            out_f.write(f"{s}_orders,{label_str}\n")

        batch_sessions.clear()
        batch_aids.clear()
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
                obj = json.loads(line)
                s = int(obj["session"])
                ev = obj["events"]
                ev.sort(key=lambda x: x["ts"])
                aids = [int(e["aid"]) for e in ev]
                unique_events = list(dict.fromkeys(aids))
                history_set = set(unique_events)

                candidates = Counter()
                for aid in unique_events[::-1]:
                    candidates[aid] += 10
                    neigh = covisitation_dict.get(aid)
                    if neigh is not None:
                        for b in neigh:
                            if b not in history_set:
                                candidates[b] += 1

                top_candidates = [aid for aid, _ in candidates.most_common(CFG.TOP_N)]
                if not top_candidates:
                    top_candidates = top_50_list[:CFG.TOP_N]

                batch_sessions.append(s)
                for aid in top_candidates:
                    batch_aids.append(aid)
                    batch_in_hist.append(1 if aid in history_set else 0)
                    batch_pop.append(pop.get(aid, 0))
                batch_offsets.append(len(batch_aids))

                if len(batch_sessions) >= 20_000:
                    flush(out_f)

        flush(out_f)

    print("Submission saved to submission.csv")

if __name__ == "__main__":
    inference()
