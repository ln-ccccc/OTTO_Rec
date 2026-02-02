# OTTO 推荐系统迭代日志 (Experiment Log)

本文档记录了从零构建 OTTO 推荐系统的演进过程，以及关键技术决策背后的思考。

---

## 📖 赛题背景与目标 (Business Understanding)

### 我们要预测什么？

用户在电商网站留下了历史行为（Session），我们需要预测该用户在未来（测试集时间段）会发生的三种行为：

1. **Clicks**: 会点击哪些商品？（浏览兴趣）
2. **Carts**: 会把哪些商品加入购物车？（购买意向）
3. **Orders**: 会最终购买哪些商品？（转化事实）

### Submission 格式要求

对测试集中的每一个 `session_id`，输出三行预测结果：

```csv
session_type,labels
session_1_clicks,item_A item_B ... (Top 20)
session_1_carts,item_C item_D ... (Top 20)
session_1_orders,item_E item_F ... (Top 20)
```

**注意**：同一个用户，点击、加购、购买的商品列表通常是不同的。

---

## 🚀 迭代记录

### v1.0: 极简 Baseline (MVP)

**架构**: 单路召回 + 单模型预测
**状态**: ❌ 已废弃 (精度低，逻辑粗糙)

#### 核心逻辑

1. **召回 (Recall)**:
   - 只使用了 `Click-Click` 共现矩阵（看了又看）。
   - 假设：只要是相关的商品，用户都会感兴趣。
2. **标签 (Label)**:
   - **混用标签**：只要用户对商品有任何行为（点/加/买），Label 都设为 1。
3. **模型 (Ranking)**:
   - 训练**一个** XGBoost 模型。
   - 模型学习的是“用户是否感兴趣（General Interest）”。
4. **预测 (Inference)**:
   - 用同一个模型打分。
   - Clicks/Carts/Orders 三个任务的预测结果是**完全一样**的（复制三份）。

#### 为什么升级？

- **问题 1**: 召回率低。只用点击数据，捕捉不到“加购后购买”或“搭配购买”的强关联。
- **问题 2**: 无法区分意图。模型分不清“想点”和“想买”。给想买手机壳的用户推荐了手机（点击率高但不会买），反之亦然。

---

### v2.0: 进阶架构 (Current)

**架构**: 三路召回 + 三模型分治
**状态**: ✅ 当前版本 

#### 改进点 1: 多路召回 (Multi-channel Recall)

不再只看“点击”，而是引入了三种矩阵：

1. **Click-Click**: 浏览兴趣漂移（权重=1）。
2. **Cart-Order**: 强转化关联（权重=3）。
3. **Buy-Buy**: 搭配购买（权重=5）。
   *效果*：召回商品的覆盖面更广，且通过加权让强关联商品更容易进入 Top 50。

#### 改进点 2: 标签拆分 (Label Splitting)

在构建训练集时，不再只打一个 `label`，而是打三个：

- `label_click`: 未来是否点击？
- `label_cart`: 未来是否加购？
- `label_order`: 未来是否购买？

#### 改进点 3: 分模型训练 (Split Ranking)

训练了三个“专家模型”：

1. `xgb_click.model`: 专门预测点击概率。
2. `xgb_cart.model`: 专门预测加购概率。
3. `xgb_order.model`: 专门预测购买概率。
   *效果*：

- Click 模型可能更看重 `pop_count`（大众热点）。
- Order 模型可能更看重 `in_history`（复购）或 Cart-Order 矩阵的权重。
- 最终输出的 Clicks/Carts/Orders 预测列表不再千篇一律。

---

## 未来规划 (v3.0+)

如果还要继续提分，可以尝试：

1. **特征工程升级**:
   - 引入 `user_buy_rate` (用户购买率)。
   - 引入 `item_conversion_rate` (商品转化率)。
   - 引入 `matrix_score` (该商品在召回矩阵里的得分)。
2. **模型升级**:
   - 尝试 `CatBoost` (处理类别特征更强)。
   - 尝试 `DeepFM` (处理稀疏特征)。
3. **验证策略**:
   - 实现严格的 `GroupKFold` 交叉验证，确保线下分数 (CV) 能对齐线上 (LB)。
