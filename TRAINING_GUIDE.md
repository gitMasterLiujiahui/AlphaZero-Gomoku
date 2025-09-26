# AlphaZero-Gomoku 训练指南（CPU 优化版）

本指南面向仅使用 CPU 的环境，目标是在约 6 小时内完成若干轮自我对弈训练，得到可用的融合模型。

## 推荐参数（CPU）
- **并行进程 `num_workers=2`**：兼顾速度与资源占用
- **每迭代自我对弈局数 `games_per_iteration=15`**：控制生成数据量
- **训练轮数 `epochs=2`**：快速过一遍数据
- 其他建议：`batch_size=256`（若内存吃紧，降到 128），`learning_rate=1e-3`

## 快速上手
```bash
python training.py
```
或自定义：
```python
from training import SelfPlayTrainer

trainer = SelfPlayTrainer(
    model_path=None,
    device="cpu",
    learning_rate=1e-3,
    batch_size=256,
)

for it in range(6):  # 约 6 小时内完成若干迭代（依据机器性能）
    games = trainer.generate_training_data(num_games=15)  # games_per_iteration
    trainer.prepare_training_data(games, train_ratio=0.85)
    for _ in range(2):  # epochs
        trainer.train_epoch()
        trainer.validate()
    # 可选评估
    stats = trainer.evaluate_model(num_games=10)
    print("Eval:", stats)
    # 自动保存
    trainer.save_model(f"models/alphazero_gomoku_iter_{it+1}.pth")
```

## 融合后训练要点
- 推理阶段使用融合 MCTS：在 `ai_agent.py` 中已集成 BG 规划增强。
- 训练阶段网络结构不变；数据来自自我对弈对局（策略分布与终局价值）。
- 建议减少 MCTS `simulations` 以提升自我对弈速度（例如 `medium` 难度：100 次）。

## 资源优化建议
- 关闭可视化与多余日志写入。
- 降低模型残差块数量（已为轻量配置）。
- 数据生成与训练交替执行，避免过长等待。

## 检查点与恢复
- 默认保存于 `models/`。
- 优先加载 `alphazero_gomoku_best.pth`，否则加载最新迭代模型。

## 常见问题
- 速度太慢：降低 `simulations` 或 `games_per_iteration`；调小 `planner_steps`。
- 内存不足：减小 `batch_size`；分批保存统计到 `data/training_stats.json`。
