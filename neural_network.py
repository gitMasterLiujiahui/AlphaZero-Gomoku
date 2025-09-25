"""
AlphaZero-Gomoku Neural Network Model
基于AlphaZero架构的五子棋神经网络模型

这个模块实现了AlphaZero-Gomoku的神经网络架构，包括：
- 残差网络结构
- 策略网络和价值网络
- 自动模型保存和加载
- 预训练模型支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os
import logging


class GomokuCNN(nn.Module):
    """五子棋卷积神经网络"""
    
    def __init__(self, board_size: int = 15, channels: int = 3):
        super(GomokuCNN, self).__init__()
        
        self.board_size = board_size
        self.channels = channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * board_size * board_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # 输出层
        self.policy_head = nn.Linear(512, board_size * board_size)  # 策略网络
        self.value_head = nn.Linear(512, 1)  # 价值网络
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 输出
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return policy, value


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AlphaZeroGomokuNet(nn.Module):
    """
    AlphaZero-Gomoku神经网络
    基于AlphaZero架构的五子棋神经网络，包含残差网络、策略网络和价值网络
    """
    
    def __init__(self, board_size: int = 15, channels: int = 3, num_residual: int = 2):
        """
        初始化AlphaZero-Gomoku网络
        参数:
            board_size: 棋盘大小，默认15x15
            channels: 输入通道数，默认3（黑子、白子、空位）
            num_residual: 残差块数量，默认6个
        """
        super(AlphaZeroGomokuNet, self).__init__()
        
        self.board_size = board_size
        self.channels = channels
        self.num_residual = num_residual
        
        # 初始卷积层：将3通道输入转换为128通道特征图（降低复杂度）
        self.conv = nn.Conv2d(channels, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        
        # 残差塔：多个残差块堆叠，提取深层特征
        self.residual_tower = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_residual)
        ])
        
        # 策略头：输出每个位置的落子概率
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)  # 2个通道用于策略
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头：输出局面的价值评估
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)   # 1个通道用于价值
        self.value_fc1 = nn.Linear(board_size * board_size, 64)  # 减小隐藏层大小
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, 3, 15, 15)
        返回:
            policy: 策略输出，形状为(batch_size, 225)
            value: 价值输出，形状为(batch_size, 1)
        """
        # 初始卷积：提取基础特征
        x = F.relu(self.bn(self.conv(x)))
        
        # 残差塔：通过多个残差块提取深层特征
        for residual_block in self.residual_tower:
            x = residual_block(x)
        
        # 策略头：计算每个位置的落子概率
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)  # 展平
        policy = self.policy_fc(policy)
        
        # 价值头：评估当前局面的价值
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)     # 展平
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # 输出范围[-1, 1]
        
        return policy, value


class GomokuModel:
    """
    AlphaZero-Gomoku模型包装器
    提供模型创建、预测、保存和加载功能
    """
    
    def __init__(self, model_path: Optional[str] = None, board_size: int = 15, device: str = "cpu"):
        """
        初始化Gomoku模型
        参数:
            model_path: 预训练模型路径，如果提供则自动加载
            board_size: 棋盘大小，默认15x15
            device: 计算设备，'auto'自动选择，'cpu'或'cuda'
        """
        self.board_size = board_size
        self.model_path = model_path

        self.device = torch.device(device)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建AlphaZero-Gomoku模型
        self.model = AlphaZeroGomokuNet(board_size)
        self.model.to(self.device)
        self.model.eval()
        
        # 自动加载预训练模型（优先加载最优模型）
        self._logged_no_pretrained_once = False
        if model_path and os.path.exists(model_path):
            ok = self.load_model(model_path)
            if ok:
                self.logger.info(f"Loaded pretrained model from {model_path}")
        else:
            # 优先尝试加载最优模型
            best_model = "models/alphazero_gomoku_best.pth"
            if os.path.exists(best_model):
                ok = self.load_model(best_model)
                if ok:
                    self.logger.info(f"Loaded best model from {best_model}")
            else:
                # 如果没有最优模型，尝试加载最新模型
                latest_model = self._find_latest_model()
                if latest_model:
                    ok = self.load_model(latest_model)
                    if ok:
                        self.logger.info(f"Loaded latest model from {latest_model}")
                else:
                    if not self._logged_no_pretrained_once:
                        self.logger.info("No pretrained model found, using random initialization")
                        self._logged_no_pretrained_once = True
        
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测策略和价值
        参数:
            board_state: 棋盘状态，形状为(15, 15)或(3, 15, 15)
        返回:
            policy: 策略输出，每个位置的落子概率
            value: 价值输出，当前局面的价值评估
        """
        # 转换为3通道张量
        if board_state.ndim == 2:
            # 单通道棋盘状态，转换为3通道
            tensor = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
            tensor[0] = (board_state == 1).astype(np.float32)  # 黑子通道
            tensor[1] = (board_state == 2).astype(np.float32)  # 白子通道
            tensor[2] = (board_state == 0).astype(np.float32)  # 空位通道
        else:
            tensor = board_state.astype(np.float32)
        
        # 添加批次维度并转换为PyTorch张量
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.model(tensor)
            
            # 转换为numpy数组
            policy = policy.cpu().numpy().flatten()
            value = value.cpu().numpy().item()
            
            # 应用softmax到策略，确保概率和为1
            policy = self._softmax(policy)
            
        return policy, value
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax函数"""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_move_probabilities(self, board_state: np.ndarray, valid_moves: list) -> np.ndarray:
        """获取有效移动的概率分布"""
        policy, _ = self.predict(board_state)
        
        # 只保留有效移动的概率
        valid_probs = np.zeros(len(valid_moves))
        for i, (row, col) in enumerate(valid_moves):
            move_idx = row * self.board_size + col
            valid_probs[i] = policy[move_idx]
        
        # 归一化
        if np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
        
        return valid_probs
    
    def _find_latest_model(self) -> Optional[str]:
        """
        查找最新的训练模型
        返回: 最新模型路径，如果没有则返回None
        """
        models_dir = "models"
        if not os.path.exists(models_dir):
            return None
        
        # 查找所有模型文件
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith("alphazero_gomoku_") and file.endswith(".pth"):
                model_files.append(os.path.join(models_dir, file))
        
        if not model_files:
            return None
        
        # 按修改时间排序，返回最新的
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_files[0]
    
    def save_model(self, filepath: str):
        """
        保存模型到指定路径
        参数:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型状态和元数据
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': 'alphazero_gomoku',
            'board_size': self.board_size,
            'device': str(self.device)
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        从指定路径加载模型
        参数:
            filepath: 模型文件路径
        返回:
            bool: 是否加载成功
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            # 严格关闭，避免结构变化导致报错
            missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.eval()
            if missing or unexpected:
                self.logger.warning(f"Loaded with mismatches. missing={len(missing)}, unexpected={len(unexpected)}")
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}")
            return False
    
    def auto_save_model(self, iteration: int = None, suffix: str = ""):
        """
        自动保存模型
        参数:
            iteration: 迭代次数，用于生成文件名
            suffix: 文件名后缀
        """
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        if iteration is not None:
            filename = f"alphazero_gomoku_iter_{iteration}{suffix}.pth"
        else:
            filename = f"alphazero_gomoku_auto{suffix}.pth"
        
        filepath = os.path.join(models_dir, filename)
        self.save_model(filepath)
        return filepath
    
    def train_mode(self):
        """切换到训练模式"""
        self.model.train()
    
    def eval_mode(self):
        """切换到评估模式"""
        self.model.eval()


def create_pretrained_model(board_size: int = 15) -> GomokuModel:
    """
    创建预训练模型
    参数:
        board_size: 棋盘大小
    返回:
        GomokuModel: 预训练模型实例
    """
    # 尝试加载预训练模型
    pretrained_path = f"models/pretrained_gomoku_{board_size}x{board_size}.pth"
    
    if os.path.exists(pretrained_path):
        model = GomokuModel(pretrained_path, board_size)
        print(f"Loaded pretrained model from {pretrained_path}")
    else:
        # 如果没有预训练模型，创建新模型并尝试加载最新训练模型
        model = GomokuModel(None, board_size)
        print("No pretrained model found, using latest trained model or random initialization")
    
    return model


def test_model():
    """测试AlphaZero-Gomoku模型"""
    print("Testing AlphaZero-Gomoku Model...")
    
    # 创建模型
    model = GomokuModel()
    
    # 创建测试棋盘
    board_state = np.zeros((15, 15), dtype=int)
    board_state[7, 7] = 1  # 中心放一个黑子
    
    # 预测
    policy, value = model.predict(board_state)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value}")
    print(f"Policy sum: {np.sum(policy)}")
    
    # 测试有效移动概率
    valid_moves = [(7, 6), (7, 8), (6, 7), (8, 7)]
    probs = model.get_move_probabilities(board_state, valid_moves)
    
    print(f"Valid move probabilities: {probs}")
    
    # 测试模型保存和加载
    test_path = "models/test_model.pth"
    model.save_model(test_path)
    print(f"Model saved to {test_path}")
    
    # 创建新模型并加载
    new_model = GomokuModel()
    if new_model.load_model(test_path):
        print("Model loaded successfully!")
    
    # 清理测试文件
    if os.path.exists(test_path):
        os.remove(test_path)
        print("Test file cleaned up")
    
    print("AlphaZero-Gomoku model test completed!")


if __name__ == "__main__":
    test_model()
