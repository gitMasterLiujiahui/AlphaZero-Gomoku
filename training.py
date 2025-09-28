"""
完善版训练脚本（CPU 友好，融合模型）

阶段一：自我对弈数据生成（AlphaZeroGomokuAI）
阶段二：模型训练与优化（数据加载、数据增强、调度、早停、梯度裁剪）
阶段三：模型评估与验证（胜率与损失）

保存策略：
- 每迭代保存 models/alphazero_gomoku_iter_{i}.pth
- 维护 models/alphazero_gomoku_best.pth（基于验证指标）
- 最终保存 models/alphazero_gomoku_final.pth

注：本脚本专为 CPU 运行优化，合理控制对弈数量与 MCTS 模拟次数。
"""

import os
import math
import random
import copy
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gomoku_board import GomokuBoard
from ai_agent import AlphaZeroGomokuAI
from neural_network import GomokuModel


# ----------------------------
# 数据增强：棋盘的 8 种对称变换
# ----------------------------

def _idx_to_rc(idx: int, n: int = 15) -> Tuple[int, int]:
    return idx // n, idx % n

def _rc_to_idx(r: int, c: int, n: int = 15) -> int:
    return r * n + c

def _transform_planes(planes: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    # planes: (3, H, W)
    x = planes
    for _ in range(k_rot % 4):
        x = np.rot90(x, k=1, axes=(1, 2))
    if flip:
        x = np.flip(x, axis=2)  # 水平翻转
    return x.copy()

def _transform_index(idx: int, k_rot: int, flip: bool, n: int = 15) -> int:
    r, c = _idx_to_rc(idx, n)
    # 将 (r,c) 应用相同的变换
    # 旋转 90 度： (r, c) -> (c, n-1-r)
    for _ in range(k_rot % 4):
        r, c = c, n - 1 - r
    if flip:
        c = n - 1 - c
    return _rc_to_idx(r, c, n)

def augment_sample(planes: np.ndarray, move_idx: int) -> List[Tuple[np.ndarray, int]]:
    out = []
    for k_rot in range(4):
        for flip in (False, True):
            x = _transform_planes(planes, k_rot, flip)
            y = _transform_index(move_idx, k_rot, flip)
            out.append((x, y))
    return out


# ----------------------------
# 简单经验缓冲区
# ----------------------------

class SimpleReplay:
    def __init__(self):
        self.states: List[np.ndarray] = []      # (3, 15, 15)
        self.move_indices: List[int] = []       # 0..224
        self.players: List[int] = []            # 1 or 2
        self.outcomes: List[int] = []           # -1/0/1

    def add(self, planes: np.ndarray, move_idx: int, player: int):
        self.states.append(planes.astype(np.float32))
        self.move_indices.append(int(move_idx))
        self.players.append(int(player))

    def finalize_with_winner(self, winner: Optional[int]):
        for p in self.players:
            if winner is None:
                self.outcomes.append(0)
            else:
                self.outcomes.append(1 if p == winner else -1)

    def __len__(self) -> int:
        return len(self.states)


# ----------------------------
# Dataset 与 DataLoader
# ----------------------------

class GomokuSelfPlayDataset(Dataset):
    def __init__(self, replay: SimpleReplay, use_augmentation: bool = True, augment_ratio: float = 0.5):
        self.samples: List[Tuple[np.ndarray, int, float]] = []  # (planes, move_idx, value)
        n = len(replay)
        indices = list(range(n))
        # 将原始样本纳入
        for i in indices:
            planes = replay.states[i]
            move_idx = replay.move_indices[i]
            value = float(replay.outcomes[i])
            self.samples.append((planes, move_idx, value))
        # 数据增强：按比例对样本做 8 倍对称增强（子集）
        if use_augmentation and n > 0:
            aug_count = int(n * augment_ratio)
            chosen = random.sample(indices, k=max(1, aug_count))
            for i in chosen:
                planes = replay.states[i]
                move_idx = replay.move_indices[i]
                value = float(replay.outcomes[i])
                for aug_planes, aug_idx in augment_sample(planes, move_idx):
                    self.samples.append((aug_planes, aug_idx, value))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        planes, move_idx, value = self.samples[idx]
        x = torch.from_numpy(planes)            # (3,15,15)
        y_policy = torch.tensor(move_idx, dtype=torch.long)
        y_value = torch.tensor([value], dtype=torch.float32)
        return x, y_policy, y_value


# ----------------------------
# 自我对弈 + 评价
# ----------------------------

def play_one_game(ai_black: AlphaZeroGomokuAI, ai_white: AlphaZeroGomokuAI, 
                  step_timeout: float = 10.0, game_timeout: float = 300.0) -> Tuple[SimpleReplay, int]:
    """
    进行一局自我对弈（带超时保护）
    参数:
        ai_black: 黑子AI
        ai_white: 白子AI
        step_timeout: 单步超时时间（秒）
        game_timeout: 整局超时时间（秒）
    """
    board = GomokuBoard()
    buf = SimpleReplay()
    start_moves = board.get_move_count()
    game_start_time = time.time()
    step_count = 0
    slow_steps = 0  # 慢步计数

    while not board.game_over:
        current_time = time.time()
        game_elapsed = current_time - game_start_time
        
        # 整局超时保护
        if game_elapsed > game_timeout:
            print(f"\r    ├── 对局超时: {step_count}步 [总用时:{game_elapsed:.1f}s] ⚠️", end="")
            break
            
        current_player = board.current_player
        ai = ai_black if current_player == GomokuBoard.BLACK else ai_white
        
        # 单步超时保护
        step_start = time.time()
        try:
            move = ai.get_move(board)
            step_elapsed = time.time() - step_start
            
            # 检查是否超时
            if step_elapsed > step_timeout:
                print(f"\r    ├── 步{step_count+1}超时: {step_elapsed:.1f}s > {step_timeout}s ⚠️", end="")
                # 使用随机移动继续
                valid_moves = board.get_valid_moves()
                if valid_moves:
                    move = random.choice(valid_moves)
                else:
                    break
            elif step_elapsed > 3.0:
                slow_steps += 1
                print(f"\r    ├── 对局{step_count+1}: 步数:{step_count+1} [总用时:{game_elapsed:.1f}s] 慢步警告⚠️", end="")
            else:
                print(f"\r    ├── 对局{step_count+1}: 步数:{step_count+1} [总用时:{game_elapsed:.1f}s]", end="")
                
        except Exception as e:
            print(f"\r    ├── 步{step_count+1}异常: {str(e)[:30]}... ⚠️", end="")
            # 使用随机移动继续
            valid_moves = board.get_valid_moves()
            if valid_moves:
                move = random.choice(valid_moves)
            else:
                break
        
        if move is None:
            break
            
        # 记录当前状态
        planes = board.get_board_tensor()
        move_idx = move[0] * board.BOARD_SIZE + move[1]
        buf.add(planes, move_idx, current_player)
        board.make_move(move[0], move[1])
        step_count += 1

    buf.finalize_with_winner(board.winner)
    game_len = board.get_move_count() - start_moves
    
    # 显示最终结果
    final_time = time.time() - game_start_time
    winner_text = "玩家1胜" if board.winner == GomokuBoard.BLACK else "玩家2胜" if board.winner == GomokuBoard.WHITE else "平局"
    print(f"\r    ├── 对局完成: {game_len}步 {winner_text} [用时:{final_time:.1f}s]")
    
    return buf, game_len


def evaluate_model(current: GomokuModel, baseline: Optional[GomokuModel], device: torch.device,
                   games: int = 8, eval_difficulty: str = "easy", eval_num_sim: int = 60,
                   eval_plans: int = 2) -> Dict[str, Any]:
    # 如果没有基线，使用随机着法作为对手
    class RandomAgent:
        def get_move(self, board: GomokuBoard):
            v = board.get_valid_moves()
            return random.choice(v) if v else None

    black_is_current = True
    wins = 0
    losses = 0
    draws = 0

    for g in range(games):
        board = GomokuBoard()
        color_a = GomokuBoard.BLACK if black_is_current else GomokuBoard.WHITE
        if baseline is None:
            ai_a = AlphaZeroGomokuAI(color_a, difficulty=eval_difficulty, device=str(device), planner_steps=eval_plans)
            ai_a.params["num_simulations"] = eval_num_sim
            ai_b = RandomAgent()
        else:
            ai_a = AlphaZeroGomokuAI(color_a, difficulty=eval_difficulty, device=str(device), planner_steps=eval_plans)
            other_color = GomokuBoard.WHITE if color_a == GomokuBoard.BLACK else GomokuBoard.BLACK
            ai_b = AlphaZeroGomokuAI(other_color, difficulty=eval_difficulty, device=str(device), planner_steps=eval_plans)
            ai_b.params["num_simulations"] = eval_num_sim
            ai_b.model = baseline
        ai_a.model = current

        while not board.game_over:
            current_player = board.current_player
            agent = ai_a if (current_player == GomokuBoard.BLACK if black_is_current else current_player == GomokuBoard.WHITE) else ai_b
            move = agent.get_move(board)
            if move is None:
                break
            board.make_move(*move)

        if board.winner is None:
            draws += 1
        else:
            if board.winner == color_a:
                wins += 1
            else:
                losses += 1

        # 交替先手到下一局
        black_is_current = not black_is_current

    total = max(1, wins + losses + draws)
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total}


# ----------------------------
# 训练一个 epoch
# ----------------------------

def train_epoch(model: GomokuModel, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device,
                grad_clip: float = 1.0, epoch_index: int = 1, num_epochs: int = 1) -> float:
    model.train_mode()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0.0
    batches = 0

    total_batches = len(loader)
    processed = 0
    for x, y_policy, y_value in loader:
        x = x.to(device)
        y_policy = y_policy.to(device)
        y_value = y_value.to(device)

        optimizer.zero_grad()
        policy_logits, value_pred = model.model(x)
        loss_policy = ce(policy_logits, y_policy)
        loss_value = mse(value_pred, y_value)
        loss = loss_policy + loss_value
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
        processed += 1
        # Epoch级别进度条（单条）
        pct = processed / max(1, total_batches)
        bar = _render_bar(pct, width=12)
        print(f"\r    ├── 模型训练: {bar} {int(pct*100)}% (epoch {epoch_index}/{num_epochs})", end="")

    print("")
    return total_loss / max(1, batches)


def validate_epoch(model: GomokuModel, loader: DataLoader, device: torch.device, epoch_index: int = 1, num_epochs: int = 1) -> float:
    model.eval_mode()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0.0
    batches = 0
    total_batches = len(loader)
    processed = 0
    with torch.no_grad():
        for x, y_policy, y_value in loader:
            x = x.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)
            policy_logits, value_pred = model.model(x)
            loss = ce(policy_logits, y_policy) + mse(value_pred, y_value)
            total_loss += float(loss.item())
            batches += 1
            processed += 1
            # 验证进度条（单条）
            pct = processed / max(1, total_batches)
            bar = _render_bar(pct, width=12)
            print(f"\r    ├── 验证中:   {bar} {int(pct*100)}% (epoch {epoch_index}/{num_epochs})", end="")
    print("")
    return total_loss / max(1, batches)


# ----------------------------
# 进度渲染工具
# ----------------------------

def _render_bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty

def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}小时{m}分"


# ----------------------------
# 主流程
# ----------------------------

def main():
    device = torch.device("cpu")
    os.makedirs("models", exist_ok=True)

    print("🚀 开始 AlphaZero 五子棋训练")
    print("=" * 50)
    print(f"设备: {device}")
    print(f"模型目录: {os.path.abspath('models')}")

    # 加载或初始化模型（内部会优先 best -> 最新 -> 随机）
    print("📦 加载模型中...")
    model = GomokuModel(model_path=None, board_size=15, device=str(device))
    print("✅ 模型加载完成")

    # 训练配置（CPU 友好）
    iterations = 6            # 总迭代轮数
    games_per_iteration = 10  # 每轮自我对弈局数（减小以提速）
    train_split = 0.9         # 更多样本用于训练
    batch_size = 128         # 批次
    learning_rate = 8e-4
    grad_clip = 0.8

    print(f"训练配置: {iterations}轮迭代, 每轮{games_per_iteration}局对弈")
    print("=" * 50)

    # 优化与调度
    optimizer = optim.Adam(model.model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85)

    # 早停机制
    patience = 3
    best_val = math.inf
    patience_count = 0

    # 评估用基线（上一轮最佳的快照）
    best_model_snapshot: Optional[GomokuModel] = None

    t0 = time.time()


    for it in range(1, iterations + 1):
        # 阶段一：自我对弈数据生成（带时间限制）
        ai_black = AlphaZeroGomokuAI(GomokuBoard.BLACK, difficulty="medium", device=str(device), 
                                    time_limit=5.0, time_reward_factor=0.1)
        ai_white = AlphaZeroGomokuAI(GomokuBoard.WHITE, difficulty="medium", device=str(device),
                                    time_limit=5.0, time_reward_factor=0.1)
        ai_black.model = model
        ai_white.model = model

        replay = SimpleReplay()
        finished_games = 0
        total_moves_acc = 0
        # 自我对弈进度（带超时保护）
        print(f"🔄 迭代 {it}/{iterations} - 阶段一：自我对弈数据生成")
        for g in range(games_per_iteration):
            buf, glen = play_one_game(ai_black, ai_white, step_timeout=8.0, game_timeout=180.0)
            replay.states.extend(buf.states)
            replay.move_indices.extend(buf.move_indices)
            replay.players.extend(buf.players)
            replay.outcomes.extend(buf.outcomes)
            finished_games += 1
            total_moves_acc += glen
            pg_pct = finished_games / games_per_iteration
            bar = _render_bar(pg_pct, width=20)
            avg_moves = int(total_moves_acc/max(1,finished_games))
            print(f"\r    ├── 自我对弈: {bar} {int(pg_pct*100)}% ({finished_games}/{games_per_iteration}局) 平均步数≈{avg_moves}", end="")
        print("")

        # 阶段二：数据加载与训练（含增强与验证）
        dataset = GomokuSelfPlayDataset(replay, use_augmentation=True, augment_ratio=0.35)
        if len(dataset) < 8:
            print("自我对弈样本过少，跳过本轮训练。")
            continue
        # 划分训练/验证
        n = len(dataset)
        idx = list(range(n))
        random.shuffle(idx)
        split = int(n * train_split)
        train_idx = idx[:split]
        val_idx = idx[split:]

        class Subset(Dataset):
            def __init__(self, base: Dataset, indices: List[int]):
                self.base = base
                self.indices = indices
    
            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                return self.base[self.indices[i]]

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        # 训练若干 epoch（这里每轮 2 次），每个epoch显示单条进度条
        train_losses = []
        val_losses = []
        epochs = 2
        for ep in range(epochs):
            # 这里在 train_epoch 内显示批次条；此处显示总体epoch进度
            train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=grad_clip, epoch_index=ep+1, num_epochs=epochs)
            val_loss = validate_epoch(model, val_loader, device, epoch_index=ep+1, num_epochs=epochs)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            ep_pct = (ep + 1) / epochs
            ep_bar = _render_bar(ep_pct, width=12)
            print(f"    ├── 模型训练: {ep_bar} {int(ep_pct*100)}% (epoch {ep+1}/{epochs})")
        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        # 学习率调度
        scheduler.step()

        # 保存当前迭代模型
        iter_path = os.path.join("models", f"alphazero_gomoku_iter_{it}.pth")
        model.save_model(iter_path)

        # 早停与最佳模型维护（以验证损失为准）
        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            patience_count = 0
            best_path = os.path.join("models", "alphazero_gomoku_best.pth")
            model.save_model(best_path)
            # 复制快照用于评估对局
            best_model_snapshot = GomokuModel(model_path=best_path, board_size=15, device=str(device))
        else:
            patience_count += 1

        # 阶段三：模型评估（对战评估）
        # 评估进度（按对局数显示）
        eval_games = 6
        wins = 0
        losses = 0
        draws = 0
        for eg in range(eval_games):
            # 单局评估
            stats = evaluate_model(model, best_model_snapshot, device=device, games=1)
            wins += stats.get('wins', 0)
            losses += stats.get('losses', 0)
            draws += stats.get('draws', 0)
            pct = (eg + 1) / eval_games
            bar = _render_bar(pct, width=12)
            print(f"\r    └── 模型评估: {bar} {int(pct*100)}% ({eg+1}/{eval_games})", end="")
        print("")
        total = max(1, wins + losses + draws)
        eval_stats = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total}

        # 顶部总体进度与时间估计
        overall_pct = it / iterations
        elapsed = time.time() - t0
        eta = (elapsed / max(1e-6, overall_pct)) * (1 - overall_pct)
        overall_bar = _render_bar(overall_pct, width=20)
        avg_game_len = int(total_moves_acc / max(1, finished_games))
        print(f"📊 总体进度: {overall_bar} {int(overall_pct*100)}%")
        print(f"    ⏱️  已运行: {_fmt_duration(elapsed)} | 预计剩余: {_fmt_duration(eta)}")
        print(f"\n    🔄 当前阶段: 模型训练 (迭代 {it}/{iterations})")
        print(f"    ├── 自我对弈: ✅ 完成 ({finished_games}/{games_per_iteration}局)")
        print(f"    ├── 模型训练: ✅ 完成 (epoch {epochs}/{epochs})")
        print(f"    └── 模型评估: ✅ 完成")

        # 性能指标展示（无 Elo 估计则省略增量，仅显示当前）
        wr = eval_stats.get('win_rate', 0.0) * 100.0
        print("\n    📈 性能指标:")
        print(f"    ├── 胜率: {wr:.1f}%")
        print(f"    ├── 平均游戏长度: {avg_game_len}步")
        print(f"    └── 损失值: {avg_val:.3f}")

        if patience_count >= patience:
            print("触发早停条件，结束训练。")
            break

    # 最终保存
    final_path = os.path.join("models", "alphazero_gomoku_final.pth")
    model.save_model(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
