"""
Parallel AlphaZero Training Framework
并行AlphaZero训练框架

实现高性能、可扩展的AlphaZero训练系统，包括：
- 多进程并行自我对弈
- 优先经验回放缓冲区
- 混合精度训练
- 增强的模型架构
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
import pickle
import json
from tqdm import tqdm
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process, Value, Lock
import threading
from collections import deque
import heapq
import math

from gomoku_board import GomokuBoard
from neural_network import AlphaZeroGomokuNet, GomokuModel
from ai_agent import AlphaZeroGomokuAI


class PriorityReplayBuffer:
    """
    优先经验回放缓冲区
    基于TD误差进行优先级采样
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.lock = Lock()
        
    def add(self, experience: Tuple, td_error: float):
        """添加经验到缓冲区"""
        with self.lock:
            priority = (abs(td_error) + 1e-6) ** self.alpha
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.priorities.append(priority)
            else:
                self.buffer[self.position] = experience
                self.priorities[self.position] = priority
                self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """基于优先级采样批次数据"""
        with self.lock:
            if len(self.buffer) == 0:
                return [], np.array([]), np.array([])
            
            # 计算采样概率
            priorities = np.array(self.priorities[:len(self.buffer)])
            probabilities = priorities / priorities.sum()
            
            # 重要性采样权重
            weights = (len(self.buffer) * probabilities) ** (-self.beta)
            weights = weights / weights.max()
            
            # 采样索引
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            
            # 获取批次数据
            batch = [self.buffer[i] for i in indices]
            weights = weights[indices]
            
            return batch, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新优先级"""
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                if idx < len(self.priorities):
                    self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class EnhancedGomokuDataset(Dataset):
    """
    增强的五子棋数据集
    支持数据增强和特征提取
    """
    
    def __init__(self, data: List[Tuple], augment: bool = True):
        self.data = data
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_state, move_index, value = self.data[idx]
        
        # 数据增强
        if self.augment and random.random() < 0.5:
            board_state = self._augment_board(board_state)
        
        # 增强特征表示
        enhanced_features = self._extract_enhanced_features(board_state)
        
        # 转换为张量
        board_tensor = torch.from_numpy(enhanced_features).float()
        move_tensor = torch.tensor(move_index, dtype=torch.long)
        value_tensor = torch.tensor(value, dtype=torch.float)
        
        return board_tensor, move_tensor, value_tensor
    
    def _augment_board(self, board: np.ndarray) -> np.ndarray:
        """棋盘数据增强"""
        # 随机旋转
        if random.random() < 0.5:
            board = np.rot90(board, random.randint(1, 3))
        
        # 随机镜像
        if random.random() < 0.5:
            board = np.fliplr(board)
        
        return board
    
    def _extract_enhanced_features(self, board: np.ndarray) -> np.ndarray:
        """
        提取增强特征
        包括：原始棋盘、连子数、气、威胁检测等
        """
        features = np.zeros((8, 15, 15), dtype=np.float32)
        
        # 原始特征 (3通道)
        features[0] = (board == 1).astype(np.float32)  # 黑子
        features[1] = (board == 2).astype(np.float32)  # 白子
        features[2] = (board == 0).astype(np.float32)  # 空位
        
        # 连子数特征 (2通道)
        features[3] = self._count_connections(board, 1)  # 黑子连子数
        features[4] = self._count_connections(board, 2)  # 白子连子数
        
        # 气数特征 (2通道)
        features[5] = self._count_liberties(board, 1)  # 黑子气数
        features[6] = self._count_liberties(board, 2)  # 白子气数
        
        # 威胁检测 (1通道)
        features[7] = self._detect_threats(board)
        
        return features
    
    def _count_connections(self, board: np.ndarray, player: int) -> np.ndarray:
        """计算每个位置的连子数"""
        connections = np.zeros((15, 15), dtype=np.float32)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(15):
            for col in range(15):
                if board[row, col] == player:
                    max_connections = 0
                    for dr, dc in directions:
                        count = 1
                        # 向一个方向计数
                        r, c = row + dr, col + dc
                        while (0 <= r < 15 and 0 <= c < 15 and board[r, c] == player):
                            count += 1
                            r += dr
                            c += dc
                        # 向相反方向计数
                        r, c = row - dr, col - dc
                        while (0 <= r < 15 and 0 <= c < 15 and board[r, c] == player):
                            count += 1
                            r -= dr
                            c -= dc
                        max_connections = max(max_connections, count)
                    connections[row, col] = max_connections
        
        return connections
    
    def _count_liberties(self, board: np.ndarray, player: int) -> np.ndarray:
        """计算每个位置的气数"""
        liberties = np.zeros((15, 15), dtype=np.float32)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for row in range(15):
            for col in range(15):
                if board[row, col] == player:
                    liberty_count = 0
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if 0 <= r < 15 and 0 <= c < 15 and board[r, c] == 0:
                            liberty_count += 1
                    liberties[row, col] = liberty_count
        
        return liberties
    
    def _detect_threats(self, board: np.ndarray) -> np.ndarray:
        """检测威胁位置"""
        threats = np.zeros((15, 15), dtype=np.float32)
        
        for row in range(15):
            for col in range(15):
                if board[row, col] == 0:  # 空位
                    threat_level = 0
                    for player in [1, 2]:
                        threat_level += self._calculate_threat_level(board, row, col, player)
                    threats[row, col] = threat_level
        
        return threats
    
    def _calculate_threat_level(self, board: np.ndarray, row: int, col: int, player: int) -> float:
        """计算特定位置的威胁级别"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_threat = 0
        
        for dr, dc in directions:
            count = 1  # 包含假设的棋子
            # 向一个方向计数
            r, c = row + dr, col + dc
            while (0 <= r < 15 and 0 <= c < 15 and board[r, c] == player):
                count += 1
                r += dr
                c += dc
            # 向相反方向计数
            r, c = row - dr, col - dc
            while (0 <= r < 15 and 0 <= c < 15 and board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                max_threat = max(max_threat, 1.0)
            elif count == 4:
                max_threat = max(max_threat, 0.8)
            elif count == 3:
                max_threat = max(max_threat, 0.5)
            elif count == 2:
                max_threat = max(max_threat, 0.2)
        
        return max_threat


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(out)


class EnhancedAlphaZeroNet(nn.Module):
    """
    增强的AlphaZero网络
    包含注意力机制和双价值头
    """
    
    def __init__(self, board_size: int = 15, channels: int = 8, num_residual: int = 6):
        super().__init__()
        
        self.board_size = board_size
        self.channels = channels
        self.num_residual = num_residual
        
        # 初始卷积层
        self.conv = nn.Conv2d(channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        
        # 残差塔
        self.residual_tower = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual)
        ])
        
        # 注意力机制
        self.attention = MultiHeadAttention(256 * board_size * board_size, num_heads=8)
        
        # 策略头
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 双价值头
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 长期价值头
        self.long_value_fc1 = nn.Linear(board_size * board_size, 256)
        self.long_value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # 初始卷积
        x = torch.relu(self.bn(self.conv(x)))
        
        # 残差塔
        for residual_block in self.residual_tower:
            x = residual_block(x)
        
        # 注意力机制
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1, 256 * self.board_size * self.board_size)
        x_attended = self.attention(x_flat)
        x = x_attended.view(batch_size, 256, self.board_size, self.board_size)
        
        # 策略头
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # 价值头
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # 长期价值头
        long_value = torch.relu(self.long_value_fc1(value.view(batch_size, -1)))
        long_value = torch.tanh(self.long_value_fc2(long_value))
        
        return policy, value, long_value


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.relu(x)
        return x


def self_play_worker(worker_id: int, num_games: int, result_queue: Queue, 
                    model_path: Optional[str] = None, difficulty: str = "medium"):
    """
    自我对弈工作进程
    """
    try:
        # 创建AI智能体
        ai = AlphaZeroGomokuAI(1, difficulty, model_path)
        
        games = []
        for i in range(num_games):
            try:
                # 进行自我对弈
                board = GomokuBoard()
                moves = []
                
                while not board.game_over and len(moves) < 200:
                    move = ai.get_move(board)
                    if move is None:
                        break
                    
                    moves.append((board.get_board_state().copy(), move, board.current_player))
                    board.make_move(move[0], move[1])
                
                # 记录游戏结果
                game_data = {
                    'moves': moves,
                    'winner': board.winner,
                    'game_length': len(moves)
                }
                games.append(game_data)
                
            except Exception as e:
                print(f"Worker {worker_id} game {i} error: {e}")
                continue
        
        result_queue.put((worker_id, games))
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        result_queue.put((worker_id, []))


class ParallelSelfPlayTrainer:
    """
    并行自我对弈训练器
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto",
                 learning_rate: float = 0.001, batch_size: int = 512,
                 num_workers: int = 4):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 创建增强模型
        self.model = EnhancedAlphaZeroNet(15, 8, 6)
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 优先经验回放缓冲区
        self.replay_buffer = PriorityReplayBuffer(capacity=100000)
        
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
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.logger.info("No pretrained model found, using random initialization")
    
    def generate_parallel_training_data(self, num_games: int) -> List[Dict]:
        """并行生成训练数据"""
        self.logger.info(f"Generating {num_games} self-play games using {self.num_workers} workers...")
        
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers
        
        # 创建进程和队列
        processes = []
        result_queue = Queue()
        
        # 启动工作进程
        for i in range(self.num_workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            p = Process(target=self_play_worker, 
                       args=(i, worker_games, result_queue, None, "medium"))
            p.start()
            processes.append(p)
        
        # 收集结果
        all_games = []
        for _ in range(self.num_workers):
            worker_id, games = result_queue.get()
            all_games.extend(games)
            self.training_stats['games_played'] += len(games)
            self.training_stats['total_moves'] += sum(game['game_length'] for game in games)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        self.logger.info(f"Generated {len(all_games)} games with {self.training_stats['total_moves']} total moves")
        return all_games
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 从缓冲区采样
        batch_data, weights, indices = self.replay_buffer.sample(self.batch_size)
        if not batch_data:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'long_value_loss': 0}
        
        # 创建数据集
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
                
                # 计算损失
                policy_loss = nn.CrossEntropyLoss()(policy_logits, move_indices)
                value_loss = nn.MSELoss()(value_pred.squeeze(), values)
                long_value_loss = nn.MSELoss()(long_value_pred.squeeze(), values)
                
                # 总损失
                total_batch_loss = policy_loss + 0.5 * value_loss + 0.3 * long_value_loss
            
            # 反向传播
            self.scaler.scale(total_batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 计算TD误差用于优先级更新
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
              epochs_per_iteration: int = 5):
        """训练模型"""
        self.logger.info(f"Starting parallel AlphaZero training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            self.training_stats['iterations'] = iteration + 1
            
            # 生成训练数据
            games = self.generate_parallel_training_data(games_per_iteration)
            
            # 将游戏数据添加到缓冲区
            for game in games:
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
                    self.replay_buffer.add(experience, 1.0)  # 初始TD误差
            
            # 训练多个epoch
            for epoch in range(epochs_per_iteration):
                train_loss = self.train_epoch()
                self.training_stats['training_loss'].append(train_loss)
                
                self.logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss['total_loss']:.4f}, "
                               f"Policy Loss = {train_loss['policy_loss']:.4f}, "
                               f"Value Loss = {train_loss['value_loss']:.4f}")
            
            # 更新学习率
            self.scheduler.step()
            
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
        self.logger.info("Training completed!")
    
    def evaluate_model(self, num_games: int = 20) -> Dict[str, float]:
        """评估模型"""
        self.logger.info(f"Evaluating model with {num_games} games...")
        
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(num_games):
            try:
                # 创建测试AI
                test_ai = AlphaZeroGomokuAI(1, "medium")
                board = GomokuBoard()
                
                while not board.game_over and board.get_move_count() < 200:
                    if board.current_player == 1:
                        # 使用当前模型
                        move = self._get_model_move(board)
                        if move:
                            board.make_move(move[0], move[1])
                    else:
                        # 使用测试AI
                        move = test_ai.get_move(board)
                        if move:
                            board.make_move(move[0], move[1])
                
                if board.winner == 1:
                    wins += 1
                elif board.winner == 2:
                    losses += 1
                else:
                    draws += 1
                    
            except Exception as e:
                self.logger.error(f"Evaluation game {i} error: {e}")
                continue
        
        win_rate = wins / num_games
        self.logger.info(f"Evaluation results: {wins} wins, {losses} losses, {draws} draws")
        self.logger.info(f"Win rate: {win_rate:.2%}")
        
        return {'wins': wins, 'losses': losses, 'draws': draws, 'win_rate': win_rate}
    
    def _get_model_move(self, board: GomokuBoard) -> Optional[Tuple[int, int]]:
        """使用当前模型获取移动"""
        try:
            self.model.eval()
            with torch.no_grad():
                # 获取棋盘状态
                board_state = board.get_board_state()
                
                # 提取增强特征
                dataset = EnhancedGomokuDataset([(board_state, 0, 0)], augment=False)
                features = dataset[0][0].unsqueeze(0).to(self.device)
                
                # 预测
                policy, _, _ = self.model(features)
                policy = torch.softmax(policy, dim=1).cpu().numpy().flatten()
                
                # 选择最佳移动
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    return None
                
                best_move = None
                best_prob = -1
                for move in valid_moves:
                    move_idx = move[0] * 15 + move[1]
                    if policy[move_idx] > best_prob:
                        best_prob = policy[move_idx]
                        best_move = move
                
                return best_move
                
        except Exception as e:
            self.logger.error(f"Model move error: {e}")
            return None
    
    def save_model(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'model_type': 'enhanced_alphazero_gomoku'
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file not found: {filepath}")
            return False
        
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
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建并行训练器
    trainer = ParallelSelfPlayTrainer(
        model_path=None,
        device="cpu",
        learning_rate=0.001,
        batch_size=256,  # 减小批大小以适应CPU
        num_workers=2    # 减少工作进程数
    )
    
    # 开始训练
    trainer.train(
        num_iterations=5,
        games_per_iteration=20,
        epochs_per_iteration=3
    )
