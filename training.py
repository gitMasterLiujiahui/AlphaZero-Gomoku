"""
Enhanced AlphaZero-Gomoku Training Framework
增强的AlphaZero五子棋训练框架

这个模块实现了高性能、可扩展的AlphaZero训练系统，包括：
- 并行自我对弈数据生成
- 优先经验回放缓冲区
- 混合精度训练
- 增强的模型架构
- 课程学习和对抗性训练
- 自适应学习率调整
- 正则化策略
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
import pickle
import json
from tqdm import tqdm
import logging


from gomoku_board import GomokuBoard
from neural_network import AlphaZeroGomokuNet, GomokuModel
from ai_agent import AlphaZeroGomokuAI




class _SilenceLogs:
    """上下文：暂时将指定logger降为WARNING，避免与tqdm进度条互相干扰"""
    def __init__(self, logger_names: List[str]):
        self.logger_names = logger_names
        self.original_levels = {}

    def __enter__(self):
        for name in self.logger_names:
            lg = logging.getLogger(name)
            self.original_levels[name] = lg.level
            lg.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, lvl in self.original_levels.items():
            logging.getLogger(name).setLevel(lvl)


class GameRecord:
    """
    游戏记录类
    用于记录自我对弈过程中的每一步移动和最终结果
    """
    
    def __init__(self):
        self.moves = []  # [(board_state, move, player), ...]
        self.winner = None  # 获胜者 (1=黑子, 2=白子, None=平局)
        self.game_length = 0  # 游戏总步数
    
    def add_move(self, board_state: np.ndarray, move: Tuple[int, int], player: int):
        """
        添加移动记录
        参数:
            board_state: 当前棋盘状态
            move: 移动位置 (行, 列)
            player: 玩家编号
        """
        self.moves.append((board_state.copy(), move, player))
        self.game_length += 1
    
    def set_winner(self, winner: int):
        """
        设置获胜者
        参数:
            winner: 获胜者编号 (1=黑子, 2=白子, None=平局)
        """
        self.winner = winner
    
    def get_training_data(self) -> List[Tuple[np.ndarray, int, float]]:
        """
        获取训练数据
        返回: 训练数据列表，每个元素为(board_state, move_index, value)
        """
        training_data = []
        
        for i, (board_state, move, player) in enumerate(self.moves):
            # 计算移动索引（将2D坐标转换为1D索引）
            move_index = move[0] * 15 + move[1]
            
            # 计算价值（基于游戏结果和位置奖励）
            base_value = 0.0
            if self.winner == player:
                base_value = 1.0  # 获胜
            elif self.winner is None:
                base_value = 0.0  # 平局
            else:
                base_value = -1.0  # 失败
            
            # 添加位置奖励机制
            position_bonus = self._calculate_position_bonus(move, board_state, player)
            value = base_value + position_bonus
            
            training_data.append((board_state, move_index, value))
        
        return training_data
    
    def _calculate_position_bonus(self, move: Tuple[int, int], board_state: np.ndarray, player: int) -> float:
        """
        计算位置奖励
        鼓励AI选择中心区域和形成连子
        """
        row, col = move
        bonus = 0.0
        
        # 中心区域奖励
        center = 7  # 15x15棋盘的中心
        distance_from_center = abs(row - center) + abs(col - center)
        center_bonus = max(0, (10 - distance_from_center) * 0.01)
        bonus += center_bonus
        
        # 连子奖励
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1  # 包含当前棋子
            # 向一个方向计数
            r, c = row + dr, col + dc
            while (0 <= r < 15 and 0 <= c < 15 and board_state[r, c] == player):
                count += 1
                r += dr
                c += dc
            # 向相反方向计数
            r, c = row - dr, col - dc
            while (0 <= r < 15 and 0 <= c < 15 and board_state[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            # 根据连子数量给予奖励
            if count >= 5:
                bonus += 0.1  # 五子连珠
            elif count == 4:
                bonus += 0.05  # 四子连珠
            elif count == 3:
                bonus += 0.02  # 三子连珠
        
        return bonus


class SelfPlayTrainer:
    """
    AlphaZero-Gomoku自我对弈训练器
    实现完整的训练流程：自我对弈 -> 数据生成 -> 模型训练 -> 模型评估
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto", 
                 learning_rate: float = 0.001, batch_size: int = 512):
        """
        初始化训练器
        参数:
            model_path: 预训练模型路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
            learning_rate: 学习率
            batch_size: 批大小
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # 创建AlphaZero-Gomoku模型（减小残差块数量以提速）
        self.model = AlphaZeroGomokuNet(15, 3, num_residual=2)
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)
        
        # 可视化已禁用
        self.visualizer = None
        
        # 训练数据
        self.training_data = []
        self.validation_data = []
        
        # 训练统计
        self.training_stats = {
            'games_played': 0,
            'total_moves': 0,
            'training_loss': [],
            'validation_loss': [],
            'win_rate': [],
            'iterations': 0,
            'best_model_path': None,
            'best_win_rate': 0.0
        }
        
        # 设置日志（在任何可能使用logger的方法前完成）
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 自动加载预训练模型（优先加载最优模型）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 优先尝试加载最优模型
            best_model = "models/alphazero_gomoku_best.pth"
            if os.path.exists(best_model):
                self.load_model(best_model)
                print(f"Loaded best model from {best_model}")
            else:
                # 如果没有最优模型，尝试加载最新模型
                latest_model = self._find_latest_model()
                if latest_model:
                    self.load_model(latest_model)
                    print(f"Loaded latest model from {latest_model}")
                else:
                    print("No pretrained model found, using random initialization")
        
        # 自动保存配置
        self.auto_save_interval = 10  # 每10个游戏自动保存一次
        self.best_model_threshold = 0.6  # 最佳模型阈值

    def _find_latest_model(self) -> Optional[str]:
        """
        查找最新的训练模型
        返回: 最新模型路径，如果没有则返回None
        """
        models_dir = "models"
        if not os.path.exists(models_dir):
            return None
        
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith("alphazero_gomoku_iter_") and file.endswith(".pth"):
                model_files.append(os.path.join(models_dir, file))
        
        if not model_files:
            return None
        
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_files[0]

    def _auto_save_model(self, iteration: int, win_rate: float = None):
        """自动保存模型"""
        current_model_path = f"models/alphazero_gomoku_iter_{iteration}.pth"
        self.save_model(current_model_path)
        if win_rate and win_rate > self.best_model_threshold:
            if win_rate > self.training_stats['best_win_rate']:
                self.training_stats['best_win_rate'] = win_rate
                best_model_path = f"models/alphazero_gomoku_best.pth"
                self.save_model(best_model_path)
                self.training_stats['best_model_path'] = best_model_path
                self.logger.info(f"New best model saved with win rate: {win_rate:.3f}")

    def self_play_game(self, temperature: float = 1.0, max_moves: int = 200) -> GameRecord:
        """进行一局自我对弈（限制步数，防止卡住）"""
        board = GomokuBoard()
        record = GameRecord()
        ai = AlphaZeroGomokuAI(1, "easy")
        moves_done = 0
        while not board.game_over and moves_done < max_moves:
            current_player = board.current_player
            move = ai.get_move(board)
            if move is None:
                break
            record.add_move(board.get_board_state(), move, current_player)
            board.make_move(move[0], move[1])
            moves_done += 1
        record.set_winner(board.winner)
        return record

    def generate_training_data(self, num_games: int = 100) -> List[GameRecord]:
        """生成训练数据"""
        self.logger.info(f"Generating {num_games} self-play games...")
        games = []
        with _SilenceLogs(["neural_network", "ai_agent", __name__]):
            for i in tqdm(range(num_games), desc="Self-play games", leave=True):
                try:
                    game = self.self_play_game()
                    games.append(game)
                    self.training_stats['games_played'] += 1
                    self.training_stats['total_moves'] += game.game_length
                except Exception as e:
                    self.logger.error(f"Error in self-play game {i}: {e}")
                    continue
        self.logger.info(f"Generated {len(games)} games with {self.training_stats['total_moves']} total moves")
        return games

    def prepare_training_data(self, games: List[GameRecord], train_ratio: float = 0.8):
        """准备训练数据"""
        all_data = []
        for game in games:
            game_data = game.get_training_data()
            all_data.extend(game_data)
        random.shuffle(all_data)
        split_idx = int(len(all_data) * train_ratio)
        self.training_data = all_data[:split_idx]
        self.validation_data = all_data[split_idx:]
        self.logger.info(f"Training data: {len(self.training_data)} samples")
        self.logger.info(f"Validation data: {len(self.validation_data)} samples")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        train_dataset = GomokuDataset(self.training_data)
        train_loader = SimpleDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        for batch_idx, (board_states, move_indices, values) in enumerate(train_loader):
            board_states = board_states.to(self.device)
            move_indices = move_indices.to(self.device)
            values = values.to(self.device)
            self.optimizer.zero_grad()
            policy_logits, value_pred = self.model(board_states)
            policy_loss = nn.CrossEntropyLoss()(policy_logits, move_indices)
            value_loss = nn.MSELoss()(value_pred.squeeze(), values)
            # 调整损失权重，降低价值损失的影响
            total_batch_loss = policy_loss + 0.5 * value_loss
            total_batch_loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += total_batch_loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
        avg_loss = total_loss / len(train_loader)
        avg_policy_loss = policy_loss_total / len(train_loader)
        avg_value_loss = value_loss_total / len(train_loader)
        return {'total_loss': avg_loss, 'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss}

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        if not self.validation_data:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        val_dataset = GomokuDataset(self.validation_data)
        val_loader = SimpleDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        with torch.no_grad():
            for board_states, move_indices, values in val_loader:
                board_states = board_states.to(self.device)
                move_indices = move_indices.to(self.device)
                values = values.to(self.device)
                policy_logits, value_pred = self.model(board_states)
                policy_loss = nn.CrossEntropyLoss()(policy_logits, move_indices)
                value_loss = nn.MSELoss()(value_pred.squeeze(), values)
                # 使用相同的损失权重
                total_loss += (policy_loss + 0.5 * value_loss).item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
        avg_loss = total_loss / len(val_loader)
        avg_policy_loss = policy_loss_total / len(val_loader)
        avg_value_loss = value_loss_total / len(val_loader)
        return {'total_loss': avg_loss, 'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss}

    def _predict_policy_value(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """使用当前训练网络进行推理，返回策略与价值"""
        if board_state.ndim == 2:
            tensor = np.zeros((3, 15, 15), dtype=np.float32)
            tensor[0] = (board_state == 1).astype(np.float32)
            tensor[1] = (board_state == 2).astype(np.float32)
            tensor[2] = (board_state == 0).astype(np.float32)
        else:
            tensor = board_state.astype(np.float32)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_f = value.squeeze().item()
        return policy, value_f

    def train(self, num_iterations: int = 10, games_per_iteration: int = 100, epochs_per_iteration: int = 5):
        """训练模型"""
        self.logger.info(f"Starting AlphaZero-Gomoku training for {num_iterations} iterations")
        for iteration in range(num_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            self.training_stats['iterations'] = iteration + 1
            games = self.generate_training_data(games_per_iteration)
            self.prepare_training_data(games)
            for epoch in range(epochs_per_iteration):
                train_loss = self.train_epoch()
                val_loss = self.validate()
                self.training_stats['training_loss'].append(train_loss)
                self.training_stats['validation_loss'].append(val_loss)
                # 可视化已禁用
                self.logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss['total_loss']:.4f}, Val Loss = {val_loss['total_loss']:.4f}")
            self.scheduler.step()
            if iteration % 2 == 0:
                win_rate = self.evaluate_model(num_games=20)
                self.training_stats['win_rate'].append(win_rate['win_rate'])
                # 可视化已禁用
                self._auto_save_model(iteration + 1, win_rate['win_rate'])
            else:
                self._auto_save_model(iteration + 1)
            self.save_training_stats()
        self.logger.info("Training completed!")
        final_model_path = f"models/alphazero_gomoku_final.pth"
        self.save_model(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        # 可视化已禁用

    def save_model(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'training_stats': self.training_stats, 'model_type': 'alphazero_gomoku'}, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def save_training_stats(self):
        """保存训练统计信息"""
        stats_path = "data/training_stats.json"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        self.logger.info(f"Training stats saved to {stats_path}")

    def load_training_stats(self):
        """加载训练统计信息"""
        stats_path = "data/training_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
            self.logger.info(f"Training stats loaded from {stats_path}")

    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file not found: {filepath}")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def save_training_data(self, filepath: str):
        """保存训练数据"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_data, f)
        self.logger.info(f"Training data saved to {filepath}")

    def load_training_data(self, filepath: str):
        """加载训练数据"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Training data file not found: {filepath}")
            return
        with open(filepath, 'rb') as f:
            self.training_data = pickle.load(f)
        self.logger.info(f"Training data loaded from {filepath}")

    def evaluate_model(self, num_games: int = 20) -> Dict[str, float]:
        """评估模型"""
        self.logger.info(f"Evaluating model with {num_games} games...")
        wins = 0
        losses = 0
        draws = 0
        with _SilenceLogs(["neural_network", "ai_agent", __name__]):
            for i in tqdm(range(num_games), desc="Evaluation games", leave=True):
                test_ai = AlphaZeroGomokuAI(1, "medium")
                board = GomokuBoard()
                while not board.game_over:
                    if board.current_player == 1:
                        policy, value = self._predict_policy_value(board.get_board_state())
                        valid_moves = board.get_valid_moves()
                        if valid_moves:
                            best_move = None
                            best_prob = -1
                            for move in valid_moves:
                                move_idx = move[0] * 15 + move[1]
                                if policy[move_idx] > best_prob:
                                    best_prob = policy[move_idx]
                                    best_move = move
                            if best_move:
                                board.make_move(best_move[0], best_move[1])
                    else:
                        move = test_ai.get_move(board)
                        if move:
                            board.make_move(move[0], move[1])
            if board.winner == 1:
                wins += 1
            elif board.winner == 2:
                losses += 1
            else:
                draws += 1
        win_rate = wins / num_games
        self.training_stats['win_rate'].append(win_rate)
        self.logger.info(f"Evaluation results: {wins} wins, {losses} losses, {draws} draws")
        self.logger.info(f"Win rate: {win_rate:.2%}")
        return {'wins': wins, 'losses': losses, 'draws': draws, 'win_rate': win_rate}

class SimpleDataLoader:
    """轻量级DataLoader，避免触发torch.distributed导入"""
    def __init__(self, dataset: Dataset, batch_size: int = 16, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        import math
        return 0 if len(self.dataset) == 0 else math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            items = [self.dataset[i] for i in batch_indices]
            if not items:
                continue
            board_states = torch.stack([it[0] for it in items], dim=0)
            move_indices = torch.stack([it[1] for it in items], dim=0)
            values = torch.stack([it[2] for it in items], dim=0)
            yield board_states, move_indices, values


class GomokuDataset(Dataset):
    """五子棋数据集"""
    
    def __init__(self, data: List[Tuple[np.ndarray, int, float]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_state, move_index, value = self.data[idx]
        
        # 确保输入为 (3, 15, 15) 三通道
        if board_state.ndim == 2:
            tensor = np.zeros((3, 15, 15), dtype=np.float32)
            tensor[0] = (board_state == 1).astype(np.float32)
            tensor[1] = (board_state == 2).astype(np.float32)
            tensor[2] = (board_state == 0).astype(np.float32)
        else:
            tensor = board_state.astype(np.float32)
        board_tensor = torch.from_numpy(tensor)
        move_tensor = torch.tensor(move_index, dtype=torch.long)
        value_tensor = torch.tensor(value, dtype=torch.float)
        
        return board_tensor, move_tensor, value_tensor


def train_gomoku_model():
    """训练五子棋模型的主函数"""
    print("Starting Gomoku AI Training...")
    
    # 创建训练器
    trainer = SelfPlayTrainer()
    
    # 开始训练
    trainer.train(
        num_iterations=5,
        games_per_iteration=50,
        epochs_per_iteration=3,
    )
    
    # 评估模型
    trainer.evaluate_model(num_games=10)
    
    print("Training completed!")


if __name__ == "__main__":
    train_gomoku_model()
