# AlphaZero-Gomoku 使用说明

## 项目简介

这是一个融合 AlphaZero 与 BG-Planner 的五子棋智能体与对战游戏。融合后的算法在蒙特卡洛树搜索（MCTS）的选择与模拟阶段引入规划增强，使搜索更具战略性与稳定性，且全面支持 CPU/GPU 设备配置与预训练模型加载。

## ✨ 新特性（融合版）
- **MCTS×BG-Planner 融合**：
  - 选择阶段：在 UCT 分数上加入 BG 规划评分奖励项，得到更稳健的节点评估。
  - 模拟阶段：前 k 步使用 BG-Planner 生成高质量着法序列，剩余步数采用快速走子，模拟更贴近真实对局。
- **设备可选**：`AlphaZeroGomokuAI(device="cpu"|"cuda")`，自动传递到内部神经网络。
- **向后兼容**：未传入 `device` 等参数时，默认保持旧行为（CPU，原MCTS流程）。

## 安装指南

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- Pygame 2.1+
- NumPy 1.21+

### 安装步骤
1. 克隆并进入项目
```bash
pip install -r requirements.txt
```

## 快速开始

### 启动游戏
```bash
python main.py
```
在设置菜单中选择：
- AI类型：`alphazero` 或 `bg_planner`
- 难度：`easy` / `medium` / `hard`

### 运行要点
- 若使用 GPU：确保安装了 GPU 版 PyTorch，并在 UI 中或代码中传入 `device="cuda"`。
- 模型自动加载：优先加载 `models/alphazero_gomoku_best.pth`，否则加载最新迭代模型。

## 融合算法参数
- **选择阶段奖励权重 `beta`**：控制 BG 评分加入 UCT 的强度，默认 `0.2`。
- **规划引导步数 `planner_steps`**：模拟阶段前 k 步使用 BG-Planner，默认 `5`。
- 可在创建 `AlphaZeroGomokuAI` 时传入，例如：
```python
from ai_agent import AlphaZeroGomokuAI
ai = AlphaZeroGomokuAI(player=1, difficulty="medium", device="cpu", beta=0.25, planner_steps=5)
```

## 难度参数（默认）
- `easy`: simulations=50, temperature=1.5, exploration=0.2, c_puct=1.2
- `medium`: simulations=100, temperature=0.8, exploration=0.05, c_puct=1.6
- `hard`: simulations=200, temperature=0.3, exploration=0.01, c_puct=1.8

## 训练与模型管理（简述）
- 训练入口：`training.py`
- 自动保存：迭代/最佳/最终模型保存到 `models/` 目录
- 推理加载：`GomokuModel(model_path=..., device=...)` 自动 to(device)

## 常见问题
- 无 GPU：使用 `device="cpu"`，并在 `TRAINING_GUIDE.md` 采用 CPU 优化配置。
- FPS 低：降低 `simulations` 或增大 `exploration`，或将 `planner_steps` 调小。

## 项目结构（简）
- `ai_agent.py`：融合后的 AI 与 MCTS
- `bg_planner.py`：BG-Planner（GraphNet + OpponentDQN + 知识搜索）
- `neural_network.py`：AlphaZero 策略-价值网络与包装器
- `game_ui.py`：Pygame 界面
- `training.py`：自我对弈训练
- `models/`：模型文件
- `data/`：训练统计数据
