"""
Curriculum Learning and Adversarial Training
课程学习和对抗性训练模块

实现：
- 课程学习：从简单局面开始，逐步增加难度
- 对抗性训练：与历史最佳模型进行对抗
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import List, Tuple, Dict, Any, Optional
import json
import logging
from collections import deque
import math

from gomoku_board import GomokuBoard
from ai_agent import AlphaZeroGomokuAI
from parallel_training import ParallelSelfPlayTrainer, EnhancedAlphaZeroNet


class CurriculumGenerator:
    """
    课程学习数据生成器
    从简单局面开始，逐步增加复杂度
    """
    
    def __init__(self):
        self.difficulty_levels = [
            {"name": "beginner", "max_moves": 10, "center_bias": 0.8, "randomness": 0.3},
            {"name": "intermediate", "max_moves": 20, "center_bias": 0.6, "randomness": 0.2},
            {"name": "advanced", "max_moves": 30, "center_bias": 0.4, "randomness": 0.1},
            {"name": "expert", "max_moves": 50, "center_bias": 0.2, "randomness": 0.05}
        ]
        self.current_level = 0
        self.level_progress = 0
        self.level_threshold = 100  # 每个级别需要完成的游戏数
    
    def get_current_difficulty(self) -> Dict[str, Any]:
        """获取当前难度级别"""
        return self.difficulty_levels[self.current_level]
    
    def update_progress(self, win_rate: float):
        """更新进度并可能升级"""
        self.level_progress += 1
        
        # 如果表现良好且达到阈值，升级
        if win_rate > 0.6 and self.level_progress >= self.level_threshold:
            if self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                self.level_progress = 0
                print(f"Upgraded to difficulty level: {self.difficulty_levels[self.current_level]['name']}")
    
    def generate_curriculum_game(self, ai1: AlphaZeroGomokuAI, ai2: AlphaZeroGomokuAI) -> Dict:
        """生成课程学习游戏"""
        difficulty = self.get_current_difficulty()
        board = GomokuBoard()
        moves = []
        
        # 根据难度调整AI行为
        ai1.difficulty_params["easy"]["exploration"] = difficulty["randomness"]
        ai2.difficulty_params["easy"]["exploration"] = difficulty["randomness"]
        
        move_count = 0
        max_moves = difficulty["max_moves"]
        
        while not board.game_over and move_count < max_moves:
            current_player = board.current_player
            
            # 根据难度选择AI
            if current_player == 1:
                move = ai1.get_move(board)
            else:
                move = ai2.get_move(board)
            
            if move is None:
                break
            
            # 应用中心偏向
            if random.random() < difficulty["center_bias"]:
                move = self._apply_center_bias(move, board)
            
            moves.append((board.get_board_state().copy(), move, current_player))
            board.make_move(move[0], move[1])
            move_count += 1
        
        return {
            'moves': moves,
            'winner': board.winner,
            'game_length': len(moves),
            'difficulty_level': difficulty["name"]
        }
    
    def _apply_center_bias(self, move: Tuple[int, int], board: GomokuBoard) -> Tuple[int, int]:
        """应用中心偏向，鼓励在中心区域下棋"""
        center = 7
        row, col = move
        
        # 如果不在中心区域，尝试移动到更靠近中心的位置
        if abs(row - center) > 2 or abs(col - center) > 2:
            valid_moves = board.get_valid_moves()
            center_moves = [(r, c) for r, c in valid_moves 
                           if abs(r - center) <= 2 and abs(c - center) <= 2]
            if center_moves:
                return random.choice(center_moves)
        
        return move


class AdversarialTrainer:
    """
    对抗性训练器
    与历史最佳模型进行对抗训练
    """
    
    def __init__(self, current_model_path: str, historical_models: List[str]):
        self.current_model_path = current_model_path
        self.historical_models = historical_models
        self.opponent_pool = []
        self.opponent_weights = []
        
        # 初始化对手池
        self._initialize_opponent_pool()
    
    def _initialize_opponent_pool(self):
        """初始化对手模型池"""
        for model_path in self.historical_models:
            if os.path.exists(model_path):
                try:
                    ai = AlphaZeroGomokuAI(2, "hard", model_path)
                    self.opponent_pool.append(ai)
                    self.opponent_weights.append(1.0)  # 初始权重
                except Exception as e:
                    print(f"Failed to load opponent model {model_path}: {e}")
    
    def select_opponent(self) -> AlphaZeroGomokuAI:
        """根据权重选择对手"""
        if not self.opponent_pool:
            # 如果没有历史模型，使用随机AI
            return AlphaZeroGomokuAI(2, "medium")
        
        # 根据权重选择对手
        weights = np.array(self.opponent_weights)
        weights = weights / weights.sum()
        opponent_idx = np.random.choice(len(self.opponent_pool), p=weights)
        
        return self.opponent_pool[opponent_idx]
    
    def update_opponent_weights(self, opponent_idx: int, result: float):
        """更新对手权重（结果：1=胜利，0=平局，-1=失败）"""
        if 0 <= opponent_idx < len(self.opponent_weights):
            # 如果对手太强，增加其权重；如果太弱，减少权重
            if result < 0:  # 对手获胜
                self.opponent_weights[opponent_idx] *= 1.1
            elif result > 0:  # 我们获胜
                self.opponent_weights[opponent_idx] *= 0.9
            
            # 限制权重范围
            self.opponent_weights[opponent_idx] = max(0.1, min(2.0, self.opponent_weights[opponent_idx]))
    
    def generate_adversarial_game(self, current_ai: AlphaZeroGomokuAI) -> Dict:
        """生成对抗性训练游戏"""
        opponent = self.select_opponent()
        board = GomokuBoard()
        moves = []
        
        while not board.game_over and len(moves) < 200:
            current_player = board.current_player
            
            if current_player == 1:
                move = current_ai.get_move(board)
            else:
                move = opponent.get_move(board)
            
            if move is None:
                break
            
            moves.append((board.get_board_state().copy(), move, current_player))
            board.make_move(move[0], move[1])
        
        return {
            'moves': moves,
            'winner': board.winner,
            'game_length': len(moves),
            'opponent_type': 'adversarial'
        }


class AdaptiveLearningRateScheduler:
    """
    自适应学习率调度器
    基于验证损失动态调整学习率
    """
    
    def __init__(self, optimizer, initial_lr: float = 0.001, patience: int = 5, 
                 factor: float = 0.5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.wait = 0
        self.lr_history = []
    
    def step(self, validation_loss: float):
        """根据验证损失调整学习率"""
        self.lr_history.append(validation_loss)
        
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                # 降低学习率
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                
                self.wait = 0


class RegularizationStrategies:
    """
    正则化策略
    包括标签平滑、随机权重平均等
    """
    
    @staticmethod
    def label_smoothing_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                           smoothing: float = 0.1) -> torch.Tensor:
        """标签平滑损失"""
        num_classes = predictions.size(-1)
        log_preds = torch.log_softmax(predictions, dim=-1)
        
        # 创建平滑标签
        smooth_targets = torch.zeros_like(log_preds)
        smooth_targets.fill_(smoothing / (num_classes - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1 - smoothing)
        
        return -torch.sum(smooth_targets * log_preds, dim=-1).mean()
    
    @staticmethod
    def weight_averaging(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
        """指数移动平均权重更新"""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class CurriculumAdversarialTrainer(ParallelSelfPlayTrainer):
    """
    课程学习和对抗性训练的综合训练器
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto",
                 learning_rate: float = 0.001, batch_size: int = 512,
                 num_workers: int = 4):
        
        super().__init__(model_path, device, learning_rate, batch_size, num_workers)
        
        # 课程学习组件
        self.curriculum = CurriculumGenerator()
        
        # 对抗性训练组件
        self.adversarial_trainer = AdversarialTrainer(
            model_path or "models/current_model.pth",
            self._get_historical_models()
        )
        
        # 自适应学习率调度器
        self.adaptive_scheduler = AdaptiveLearningRateScheduler(
            self.optimizer, learning_rate
        )
        
        # EMA模型用于权重平均
        self.ema_model = EnhancedAlphaZeroNet(15, 8, 6)
        self.ema_model.to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        
        # 正则化参数
        self.label_smoothing = 0.1
        self.weight_decay = 1e-4
        
    def _get_historical_models(self) -> List[str]:
        """获取历史模型列表"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            return []
        
        historical_models = []
        for file in os.listdir(models_dir):
            if file.startswith("alphazero_gomoku_iter_") and file.endswith(".pth"):
                historical_models.append(os.path.join(models_dir, file))
        
        # 按修改时间排序，取最近的几个
        historical_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return historical_models[:5]  # 保留最近5个模型
    
    def generate_curriculum_data(self, num_games: int) -> List[Dict]:
        """生成课程学习数据"""
        self.logger.info(f"Generating {num_games} curriculum games...")
        
        games = []
        current_ai = AlphaZeroGomokuAI(1, "medium")
        
        for i in range(num_games):
            try:
                # 生成课程学习游戏
                game = self.curriculum.generate_curriculum_game(current_ai, current_ai)
                games.append(game)
                
                # 更新课程进度
                if i % 10 == 0:
                    win_rate = self._calculate_recent_win_rate(games[-10:])
                    self.curriculum.update_progress(win_rate)
                    
            except Exception as e:
                self.logger.error(f"Curriculum game {i} error: {e}")
                continue
        
        return games
    
    def generate_adversarial_data(self, num_games: int) -> List[Dict]:
        """生成对抗性训练数据"""
        self.logger.info(f"Generating {num_games} adversarial games...")
        
        games = []
        current_ai = AlphaZeroGomokuAI(1, "hard")
        
        for i in range(num_games):
            try:
                # 生成对抗性游戏
                game = self.adversarial_trainer.generate_adversarial_game(current_ai)
                games.append(game)
                
            except Exception as e:
                self.logger.error(f"Adversarial game {i} error: {e}")
                continue
        
        return games
    
    def _calculate_recent_win_rate(self, games: List[Dict]) -> float:
        """计算最近的胜率"""
        if not games:
            return 0.0
        
        wins = sum(1 for game in games if game['winner'] == 1)
        return wins / len(games)
    
    def train_epoch_with_regularization(self) -> Dict[str, float]:
        """带正则化的训练epoch"""
        self.model.train()
        
        # 从缓冲区采样
        batch_data, weights, indices = self.replay_buffer.sample(self.batch_size)
        if not batch_data:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'long_value_loss': 0}
        
        # 创建数据集
        from parallel_training import EnhancedGomokuDataset
        dataset = EnhancedGomokuDataset(batch_data, augment=True)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(batch_data)), 
                               shuffle=True, num_workers=2, pin_memory=True)
        
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        long_value_loss_total = 0
        td_errors = []
        
        for batch_idx, (board_states, move_indices, values) in enumerate(dataloader):
            board_states = board_states.to(self.device, non_blocking=True)
            move_indices = move_indices.to(self.device, non_blocking=True)
            values = values.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                policy_logits, value_pred, long_value_pred = self.model(board_states)
                
                # 使用标签平滑的损失
                policy_loss = RegularizationStrategies.label_smoothing_loss(
                    policy_logits, move_indices, self.label_smoothing
                )
                value_loss = nn.MSELoss()(value_pred.squeeze(), values)
                long_value_loss = nn.MSELoss()(long_value_pred.squeeze(), values)
                
                # 总损失
                total_batch_loss = policy_loss + 0.5 * value_loss + 0.3 * long_value_loss
            
            # 反向传播
            self.scaler.scale(total_batch_loss).backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新EMA模型
            RegularizationStrategies.weight_averaging(self.model, self.ema_model)
            
            # 计算TD误差
            with torch.no_grad():
                td_error = torch.abs(value_pred.squeeze() - values).cpu().numpy()
                td_errors.extend(td_error)
            
            total_loss += total_batch_loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            long_value_loss_total += long_value_loss.item()
        
        # 更新优先级
        if td_errors:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = policy_loss_total / len(dataloader)
        avg_value_loss = value_loss_total / len(dataloader)
        avg_long_value_loss = long_value_loss_total / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'long_value_loss': avg_long_value_loss
        }
    
    def train(self, num_iterations: int = 10, games_per_iteration: int = 100, 
              epochs_per_iteration: int = 5, curriculum_ratio: float = 0.6):
        """综合训练方法"""
        self.logger.info(f"Starting curriculum-adversarial training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            self.training_stats['iterations'] = iteration + 1
            
            # 计算课程学习和对抗性训练的游戏数量
            curriculum_games = int(games_per_iteration * curriculum_ratio)
            adversarial_games = games_per_iteration - curriculum_games
            
            # 生成课程学习数据
            curriculum_data = self.generate_curriculum_data(curriculum_games)
            
            # 生成对抗性训练数据
            adversarial_data = self.generate_adversarial_data(adversarial_games)
            
            # 合并数据
            all_games = curriculum_data + adversarial_data
            
            # 将游戏数据添加到缓冲区
            for game in all_games:
                for board_state, move, player in game['moves']:
                    move_index = move[0] * 15 + move[1]
                    
                    # 计算价值
                    if game['winner'] == player:
                        value = 1.0
                    elif game['winner'] is None:
                        value = 0.0
                    else:
                        value = -1.0
                    
                    # 添加到缓冲区
                    experience = (board_state, move_index, value)
                    self.replay_buffer.add(experience, 1.0)
            
            # 训练多个epoch
            validation_losses = []
            for epoch in range(epochs_per_iteration):
                train_loss = self.train_epoch_with_regularization()
                self.training_stats['training_loss'].append(train_loss)
                
                # 计算验证损失（使用EMA模型）
                val_loss = self._validate_ema_model()
                validation_losses.append(val_loss)
                
                self.logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss['total_loss']:.4f}, "
                               f"Val Loss = {val_loss:.4f}")
            
            # 自适应学习率调整
            avg_validation_loss = np.mean(validation_losses)
            self.adaptive_scheduler.step(avg_validation_loss)
            
            # 保存模型
            self.save_model(f"models/alphazero_gomoku_iter_{iteration + 1}.pth")
            
            # 评估模型
            if iteration % 2 == 0:
                win_rate = self.evaluate_model(num_games=20)
                self.training_stats['win_rate'].append(win_rate['win_rate'])
                
                if win_rate['win_rate'] > self.training_stats['best_win_rate']:
                    self.training_stats['best_win_rate'] = win_rate['win_rate']
                    self.save_model("models/alphazero_gomoku_best.pth")
                    self.training_stats['best_model_path'] = "models/alphazero_gomoku_best.pth"
        
        # 保存最终模型
        self.save_model("models/alphazero_gomoku_final.pth")
        self.logger.info("Curriculum-adversarial training completed!")
    
    def _validate_ema_model(self) -> float:
        """使用EMA模型进行验证"""
        self.ema_model.eval()
        
        # 从缓冲区采样验证数据
        batch_data, _, _ = self.replay_buffer.sample(min(100, len(self.replay_buffer)))
        if not batch_data:
            return 0.0
        
        from parallel_training import EnhancedGomokuDataset
        dataset = EnhancedGomokuDataset(batch_data, augment=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        total_loss = 0
        with torch.no_grad():
            for board_states, move_indices, values in dataloader:
                board_states = board_states.to(self.device)
                move_indices = move_indices.to(self.device)
                values = values.to(self.device)
                
                policy_logits, value_pred, long_value_pred = self.ema_model(board_states)
                
                policy_loss = nn.CrossEntropyLoss()(policy_logits, move_indices)
                value_loss = nn.MSELoss()(value_pred.squeeze(), values)
                long_value_loss = nn.MSELoss()(long_value_pred.squeeze(), values)
                
                total_loss += (policy_loss + 0.5 * value_loss + 0.3 * long_value_loss).item()
        
        return total_loss / len(dataloader)


if __name__ == "__main__":
    # 创建课程学习和对抗性训练器
    trainer = CurriculumAdversarialTrainer(
        model_path=None,
        device="cpu",
        learning_rate=0.001,
        batch_size=256,
        num_workers=2
    )
    
    # 开始训练
    trainer.train(
        num_iterations=5,
        games_per_iteration=30,
        epochs_per_iteration=3,
        curriculum_ratio=0.6
    )
