import os

class CFG:
    # 数据路径
    RAW_DATA_PATH = "./data"
    PROCESSED_DATA_PATH = "./resources"
    
    # 验证集设置
    # 用训练集最后 7 天作为验证集（时间切分）
    # 通过环境变量 OTTO_DEBUG=1 启用小样本快速调试
    DEBUG = bool(int(os.environ.get("OTTO_DEBUG", "0")))
    
    # 动态路径：Debug 模式下使用采样的小文件
    TRAIN_FILE = "train_debug.jsonl" if DEBUG else "train.jsonl"
    TEST_FILE = "test_debug.jsonl" if DEBUG else "test.jsonl"
    
    # Co-visitation Logic
    # Number of candidates to recall
    # 增加候选数量至 100，显著提升召回率上限
    # 注意：这会使特征文件大小翻倍，训练时间变长，内存占用增加
    TOP_N = 100 
    
    # Disk Cache
    CACHE_PATH = "./resources/cache"
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)

# 行为类型映射（clicks/carts/orders -> 0/1/2）
TYPE_MAP = {'clicks': 0, 'carts': 1, 'orders': 2}
TYPE_WEIGHTS = {0: 1, 1: 6, 2: 3} # 共现强度的类型权重
