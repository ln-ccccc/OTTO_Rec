# OTTO 推荐系统实战

本文档基于 Kaggle OTTO 竞赛的 Candidate ReRank 方案（Chris Deotte 的思路），为您拆解一个完整的、工业级的 Session-based 推荐系统。

本项目采用经典的 **召回 (Recall) + 排序 (Ranking)** 两阶段架构。

本仓库当前实现以“**稳定不漏读**”为第一优先级：全流程直接读取原始 **JSONL**，并用 **pandas/numpy** 完成训练数据组装与特征处理；不再生成/依赖 parquet 产物。

---

## 1. 项目架构概览

```mermaid
graph LR
    A[原始日志 JSONL] --> B(数据预处理 ETL);
    B --> C{召回阶段 Recall};
    C -->|i2i 共现矩阵| D[候选集生成 Candidates];
    B --> E[特征工程 Features];
    D --> E;
    E --> F[排序模型 Ranking (XGBoost)];
    F --> G[最终预测 Submission];
```

### 核心模块说明

1. **数据预处理**: 清洗数据，划分训练集/验证集。
2. **召回 (Recall)**: 从海量商品（180万+）中快速筛选 Top 100 候选。
3. **特征工程**: 构造用户与商品的特征（如热度、历史交互）。
4. **排序 (Ranking)**: 使用 XGBoost 对候选商品精细打分。
5. **兜底策略**: 针对冷启动用户使用全局热品填充。

---

## 2. 详细 Pipeline 解析

### 2.1 数据预处理 (Preprocessing)

**脚本**: `src/01_preprocess.py`

推荐系统的第一步是确定训练/验证的切分方式与可复现实验设置。本项目使用 **时间切分**，并把切分阈值写入文件，供后续所有脚本统一读取，保证训练、召回、特征、推理都用同一个 cutoff。

当前实现不会把 JSONL 转成 parquet（避免“缓存文件过期/被截断/路径混用”导致的数据覆盖问题）。如果你后续追求速度，再引入 parquet 也没问题，但必须做好校验（例如统计 session 数、文件版本号）。

**关键动作**:

- **时间切分 (Time-based Split)**:
  - 推荐系统严禁使用随机切分（Random Split）。
  - 我们将训练集的时间轴切一刀：**最后 7 天**作为验证集（Validation），之前的作为训练历史。
  - 目的：模拟真实业务场景（用过去预测未来），防止数据穿越。
  - 输出：`resources/valid_cutoff.json`，包含 `max_ts` 与 `valid_cutoff`。

### 2.2 召回阶段：共现矩阵 (Co-visitation Matrix)

**脚本**: `src/02_covisitation.py`

这是本方案的核心（Recall）。由于用户是匿名的（Session），我们无法使用 User-based CF，只能使用 **Item-based CF**。

**算法逻辑**:

- **假设**: "看了又看"、"买了又买"。如果 Item A 和 Item B 经常在同一个 Session 里出现，它们就是相关的。
- **当前实现（更易复现、内存更稳）**:
  - 只取每个 session 的**最近 30 次行为**；
  - 只统计**相邻点击的转移**（A→B、B→A）作为共现强度（计数累加）；
  - 对每个商品保留 Top 20 邻居，输出 `resources/covisitation_click_click.pkl`。
- **你可以做的升级**:
  - 从“相邻转移”升级为“同 session 内两两配对 + 时间衰减权重”（效果更强，但更吃资源）；
  - 增加 `Cart-Order` / `Buy-Buy` 等多路召回，并在推理时融合。

### 2.3 候选生成 (Candidate Generation)

**脚本**: `src/03_candidates.py`

对于验证集/测试集中的每个用户 Session：

1. **历史回溯**: 首先把用户刚刚看过的商品加入候选（复购概率极大）。
2. **矩阵扩散**: 拿着用户看过的商品，去共现矩阵里查“邻居”。
   - *例子*: 用户看了 iPhone，矩阵推荐了 AirPods。
3. **截断**: 每个 Session 最终保留约 20-50 个候选商品。

训练阶段额外需要标签（监督信号）：
- ts < cutoff 的事件作为输入历史；
- ts >= cutoff 的事件作为未来目标（Ground Truth）；
- 候选商品如果出现在 Ground Truth 集合里，label=1，否则 label=0；
- 输出：`resources/candidates_train.pkl`（pandas pickle）。

### 2.4 特征工程 (Feature Engineering)

**脚本**: `src/04_features.py`

为了让排序模型知道哪个候选更好，我们需要构造特征：

1. **Item 特征**: 全局热度（Popularity）。
2. **User-Item 交互特征**: `in_history`（用户之前是否看过该商品？这是最强特征）。
3. **上下文特征**: Session 长度、活跃时间等。

当前实现的特征非常“精简”，目的是让你先吃透流程。你后续可以加：\n- 共现分数（A→B 的共现计数或加权分数）\n- 最近一次点击的时间间隔（recency）\n- Session 长度/去重长度等统计特征\n
输出：`resources/train_features.pkl`。

### 2.5 排序模型 (Ranking)

**脚本**: `src/05_train_ranker.py`

#### 为什么选择 XGBoost？
在推荐系统的排序阶段（Ranking），XGBoost (Gradient Boosting Decision Tree) 是工业界最常用的基线模型，原因如下：
1.  **处理非线性能力强**：推荐系统中的特征往往不是线性的（例如“点击次数”对“购买率”的影响不是直线的），树模型能自动学习这些非线性关系。
2.  **特征组合自动化**：不需要像逻辑回归（LR）那样手动做大量的特征交叉（Feature Crossing），树模型能自动发现“点击过A且热度大于100”这样的组合特征。
3.  **可解释性强**：可以通过 Feature Importance 明确知道哪个特征最重要（如 `in_history` 远比 `pop_count` 重要），便于业务排查。
4.  **鲁棒性**：对特征的缺失值、异常值不敏感，工程落地简单。

#### 核心参数解读：`rank:pairwise`
我们使用了 `objective='rank:pairwise'`，这是专门为排序任务设计的损失函数。

-   **Pointwise（点对点）**：像普通的二分类（Binary Classification）一样，单独预测每个商品“会被点击”的概率（0~1）。
    -   *缺点*：模型会过度关注那些容易预测的负样本（绝大多数商品都不会被点击），而忽略了正样本之间的相对顺序。
-   **Pairwise（成对）**：模型输入的是一个“样本对” `<Item A, Item B>`。
    -   如果用户点击了 A 而没点击 B，模型就学习让 `Score(A) > Score(B)`。
    -   *优点*：直接优化“相对顺序”。在推荐场景下，我们并不关心 A 的得分是 0.9 还是 0.8，我们只关心 A 是不是排在 B 前面。这与 NDCG（Normalized Discounted Cumulative Gain）等排序评估指标更一致。

#### 关键实现细节
-   **Group 信息**：XGBoost 做排序时，必须告诉模型哪些样本属于同一个 Query（在这里就是同一个 Session）。
    -   代码中 `groups = df.groupby("session").size().to_numpy()` 就是在做这件事。模型只会比较**同一个 Session 内部**的商品顺序，不会跨 Session 比较。

输出：`resources/xgb_ranker.model`。

### 2.6 推理与兜底 (Inference & Fallback)

**脚本**: `src/06_inference.py`

- 推理阶段直接逐行读取 `test.jsonl`，对每个 session 生成候选并批量送入模型打分。\n- 如果候选为空（极少见），使用“全局 Top20 热门商品”兜底，避免出现空 labels。\n- 输出：`submission.csv`。

---

## 3. 进阶学习路线 (致准算法工程师)

如果你想在面试中脱颖而出，建议基于此 Pipeline 尝试以下改进：

### 🌟 召回层 (Recall) 升级

- **Swing 算法**: 相比简单的共现，Swing 考虑了 User-Item-User 的结构，能消除“小圈子”噪音。
- **Embedding**: 使用 Word2Vec (Item2Vec) 训练商品向量，通过向量相似度召回（Faiss）。
- **Graph**: 尝试 DeepWalk 或 Node2Vec 图算法。

### 🌟 排序层 (Ranking) 升级

- **深度学习**: 尝试用 DeepFM 或 DIN (Deep Interest Network) 替换 XGBoost。
- **多目标学习 (Multi-task)**: 同时预测 Click、Cart、Order 三个目标。

### 🌟 工程优化

- **数据 IO**: 在保证不漏读的前提下，引入 parquet + 校验机制（session 数/文件 hash/版本号）。\n- **并行与缓存**: 对 item_pop、covisitation、候选结果做缓存，减少重复扫描 JSONL。\n- **GPU/加速**: 学习使用 cuDF / Faiss / RAPIDS 进行大规模加速。
- **增量学习**: 思考如何处理实时流入的新数据。

---

## 4. 运行指南

确保已安装依赖：

```bash
pip install pandas numpy xgboost scikit-learn
```

按顺序执行 Pipeline：

```bash
# 1. 预处理
python src/01_preprocess.py

# 2. 生成共现矩阵
python src/02_covisitation.py

# 3. 生成训练候选集
python src/03_candidates.py

# 4. 生成特征
python src/04_features.py

# 5. 训练排序模型
python src/05_train_ranker.py

# 6. 预测并生成提交文件 (自动处理冷启动)
python src/06_inference.py
```

### 4.1 调试模式（强烈推荐）

调试模式会只处理少量数据，确保你能快速跑通全流程：\n
```bash
OTTO_DEBUG=1 python src/01_preprocess.py
OTTO_DEBUG=1 python src/02_covisitation.py
OTTO_DEBUG=1 python src/03_candidates.py
OTTO_DEBUG=1 python src/04_features.py
OTTO_DEBUG=1 python src/05_train_ranker.py
OTTO_DEBUG=1 python src/06_inference.py
```

你在调试模式下看到的 submission 行数会远小于 Kaggle 要求，这是正常的；提交前务必关闭 `OTTO_DEBUG`，用全量数据生成 `submission.csv`。

### 4.2 关键产物（resources/）

- `valid_cutoff.json`：时间切分阈值（所有脚本共享）\n- `covisitation_click_click.pkl`：共现矩阵（召回用）\n- `item_pop.pkl`：商品热度缓存\n- `candidates_train.pkl`：训练候选与标签\n- `train_features.pkl`：训练特征\n- `xgb_ranker.model`：排序模型\n
