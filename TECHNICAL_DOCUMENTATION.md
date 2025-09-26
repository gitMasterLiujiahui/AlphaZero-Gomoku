# AlphaZero-Gomoku 技术文档（融合版）

本文面向新手，解释融合 AlphaZero 与 BG-Planner 的整体原理与实现，并配有类图与数据流图。

## 架构概览
- AlphaZero：策略-价值网络 + MCTS 搜索
- BG-Planner：GraphNet（全局规划）、OpponentDQN（对手建模）、KnowledgeSearch（启发式知识）
- 融合点：
  - 选择阶段：UCB/PUCT 分数 + β×BG_Score（节点奖励）
  - 模拟阶段：前 k 步使用 BG-Planner 引导，高质量着法；其后快速走子

## 核心类图（简化）
```
+----------------------+          +-------------------+
| AlphaZeroGomokuAI    |          | GomokuModel       |
| - model: GomokuModel |<>--------| - model: Net      |
| - bg_planner: BG...  |          | + predict()       |
| - knowledge_search   |          +-------------------+
| + get_move()         |
| + evaluate_position()|
+----------+-----------+
           |
           v
+----------------------+          +-------------------+
| MCTSNode             |          | BGPlannerAI       |
| - board              |          | - GraphNet        |
| - visits, value      |          | - OpponentDQN     |
| - ucb1(): UCT+β*BG   |          | - KnowledgeSearch |
+----------------------+          +-------------------+
```

## 数据流图（推理）
```
棋盘状态 → 神经网络预测(策略/价值) → MCTS树展开
                 ↓                       ↑
           BG_Score(启发)  ——>  选择(UCT + β×BG)
                 ↓
        模拟：前k步BG规划→余步随机→终局/评估
```

## 关键公式
- 选择分数（增强）：
  - Score = UCT + β × BG_Score
  - β（beta）默认 0.2，可在 `AlphaZeroGomokuAI(..., beta=0.2)` 调整。
- BG_Score：使用 `KnowledgeSearch._pattern_score` 的启发式评分，经 `tanh(score/10000)` 压缩，量纲对齐。

## 代码落点
- `ai_agent.py`：
  - `AlphaZeroGomokuAI.__init__(..., device=None, beta=0.2, planner_steps=5)`：设备、融合参数（向后兼容）。
  - `_select/_expand/_simulate/_backpropagate`：MCTS 四阶段。
  - `_simulate`：实现“前k步 BG 规划引导 + 随机走子 + 终局/网络评估”。
  - `MCTSNode.ucb1()`：在基础 UCT 分数上加入 `β×BG_Score`。
- `bg_planner.py`：
  - `GraphNet`/`OpponentDQN`/`KnowledgeSearch` 组成 BG-Planner。
  - `BGPlannerAI.get_move()`：在模拟引导阶段调用，以生成高质量着法序列。
- `neural_network.py`：
  - `GomokuModel(device=...)`：模型搬移到指定设备；`predict()` 在该设备推理。

## 设备与兼容性
- `AlphaZeroGomokuAI(device=None)`：未传入时默认为 `cpu`，完全兼容旧版本调用。
- 设备会传入 `GomokuModel(device=...)` 并 `to(device)`。

## 参数建议
- 对局响应更快：调小 `simulations`、`planner_steps` 或增大 `exploration`。
- 更强棋力：适当增大 `simulations`，并保留 `planner_steps≈5`。

## 训练与部署
- 训练流程不变（自我对弈→训练→评估→保存），详见 `TRAINING_GUIDE.md`。
- 部署只需在 UI 或构造函数中选择设备与难度。

## 故障排查
- 设备报错：确认传入 `device="cpu"|"cuda"`；PyTorch 是否安装匹配版本。
- 推理变慢：适当降低 `simulations` 和 `planner_steps`。
