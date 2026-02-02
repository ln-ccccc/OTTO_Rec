import json
import pickle
from collections import Counter
from pathlib import Path
from config import CFG
from split_utils import load_valid_cutoff


def load_or_build_item_pop(raw_data_path, processed_data_path):
    cache_path = Path(processed_data_path) / "item_pop.pkl"
    if cache_path.exists():
        print("Loading Item Popularity from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Building Item Popularity...")
    train_path = Path(raw_data_path) / CFG.TRAIN_FILE
    print(f"Reading from {train_path}...")
    
    pop = Counter()
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for e in obj["events"]:
                pop[e["aid"]] += 1
                
    with open(cache_path, "wb") as f:
        pickle.dump(pop, f)
    return pop
