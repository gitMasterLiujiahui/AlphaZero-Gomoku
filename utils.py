"""
Utility Functions for Gomoku AI
五子棋AI工具函数
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_config(config: Dict[str, Any], filepath: str):
    """保存配置到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: Any, filepath: str):
    """保存数据到pickle文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_data(filepath: str) -> Any:
    """从pickle文件加载数据"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def board_to_string(board: np.ndarray) -> str:
    """将棋盘转换为字符串表示"""
    symbols = {0: '.', 1: '●', 2: '○'}
    
    lines = []
    lines.append("   " + " ".join(f"{i:2d}" for i in range(board.shape[1])))
    
    for i, row in enumerate(board):
        line = f"{i:2d} " + " ".join(symbols[cell] for cell in row)
        lines.append(line)
    
    return "\n".join(lines)


def string_to_board(board_str: str) -> np.ndarray:
    """将字符串表示转换为棋盘"""
    lines = board_str.strip().split('\n')
    
    # 跳过标题行
    data_lines = lines[1:]
    
    board = np.zeros((len(data_lines), 15), dtype=int)
    
    for i, line in enumerate(data_lines):
        # 跳过行号
        cells = line.split()[1:]
        
        for j, cell in enumerate(cells):
            if cell == '●':
                board[i, j] = 1
            elif cell == '○':
                board[i, j] = 2
    
    return board


def get_board_hash(board: np.ndarray) -> str:
    """获取棋盘的哈希值"""
    return str(hash(board.tobytes()))


def rotate_board(board: np.ndarray, angle: int) -> np.ndarray:
    """旋转棋盘"""
    if angle == 90:
        return np.rot90(board, 1)
    elif angle == 180:
        return np.rot90(board, 2)
    elif angle == 270:
        return np.rot90(board, 3)
    else:
        return board.copy()


def flip_board(board: np.ndarray, axis: int) -> np.ndarray:
    """翻转棋盘"""
    return np.flip(board, axis=axis)


def augment_board(board: np.ndarray) -> List[np.ndarray]:
    """数据增强：生成棋盘的多个变体"""
    augmented = [board.copy()]
    
    # 旋转
    for angle in [90, 180, 270]:
        augmented.append(rotate_board(board, angle))
    
    # 翻转
    augmented.append(flip_board(board, 0))  # 水平翻转
    augmented.append(flip_board(board, 1))  # 垂直翻转
    
    return augmented


def calculate_game_complexity(board: np.ndarray) -> float:
    """计算游戏复杂度"""
    # 基于已下棋子数量和位置分布
    stone_count = np.sum(board != 0)
    
    if stone_count == 0:
        return 0.0
    
    # 计算棋子分布的熵
    positions = np.where(board != 0)
    if len(positions[0]) == 0:
        return 0.0
    
    # 计算位置的标准差
    row_std = np.std(positions[0])
    col_std = np.std(positions[1])
    
    # 复杂度 = 棋子数量 * 位置分布的标准差
    complexity = stone_count * (row_std + col_std) / 100.0
    
    return min(complexity, 1.0)


def get_opening_moves(board_size: int = 15) -> List[Tuple[int, int]]:
    """获取开局移动建议"""
    center = board_size // 2
    
    # 中心区域的开局移动
    opening_moves = [
        (center, center),           # 天元
        (center - 1, center),      # 上
        (center + 1, center),       # 下
        (center, center - 1),       # 左
        (center, center + 1),       # 右
        (center - 1, center - 1),  # 左上
        (center - 1, center + 1),  # 右上
        (center + 1, center - 1),  # 左下
        (center + 1, center + 1),  # 右下
    ]
    
    return opening_moves


def analyze_position(board: np.ndarray, player: int) -> Dict[str, Any]:
    """分析局面"""
    analysis = {
        'stone_count': np.sum(board != 0),
        'player_stones': np.sum(board == player),
        'opponent_stones': np.sum(board == (3 - player)),
        'empty_spaces': np.sum(board == 0),
        'threats': [],
        'recommendations': []
    }
    
    # 检查威胁
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] == 0:
                # 检查这个位置的威胁
                threat_level = check_position_threat(board, row, col, player)
                if threat_level > 0:
                    analysis['threats'].append({
                        'position': (row, col),
                        'threat_level': threat_level
                    })
    
    # 按威胁级别排序
    analysis['threats'].sort(key=lambda x: x['threat_level'], reverse=True)
    
    return analysis


def check_position_threat(board: np.ndarray, row: int, col: int, player: int) -> int:
    """检查特定位置的威胁级别"""
    threat_level = 0
    
    # 检查四个方向
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        # 检查连续棋子
        count = 1  # 包含当前假设的棋子
        
        # 向一个方向计数
        r, c = row + dr, col + dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r += dr
            c += dc
        
        # 向相反方向计数
        r, c = row - dr, col - dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r -= dr
            c -= dc
        
        # 根据连续数量计算威胁
        if count >= 5:
            threat_level += 1000
        elif count == 4:
            threat_level += 100
        elif count == 3:
            threat_level += 10
        elif count == 2:
            threat_level += 1
    
    return threat_level


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def format_memory(bytes_size: int) -> str:
    """格式化内存大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
    }
    
    # PyTorch信息
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info['cuda_available'] = False
    
    return info


def create_directory_structure():
    """创建项目目录结构"""
    directories = [
        'models',
        'data',
        'logs',
        'configs',
        'saves',
        'exports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def validate_board_state(board: np.ndarray) -> bool:
    """验证棋盘状态是否有效"""
    if not isinstance(board, np.ndarray):
        return False
    
    if board.shape != (15, 15):
        return False
    
    if not np.all(np.isin(board, [0, 1, 2])):
        return False
    
    return True


def get_board_statistics(board: np.ndarray) -> Dict[str, Any]:
    """获取棋盘统计信息"""
    stats = {
        'total_cells': board.size,
        'empty_cells': np.sum(board == 0),
        'black_stones': np.sum(board == 1),
        'white_stones': np.sum(board == 2),
        'fill_ratio': np.sum(board != 0) / board.size,
        'black_ratio': np.sum(board == 1) / board.size,
        'white_ratio': np.sum(board == 2) / board.size
    }
    
    return stats


def export_game_data(games: List[Any], filepath: str, format: str = 'json'):
    """导出游戏数据"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(games, f, indent=2, ensure_ascii=False, default=str)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(games, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def import_game_data(filepath: str, format: str = 'json') -> List[Any]:
    """导入游戏数据"""
    if not os.path.exists(filepath):
        return []
    
    if format == 'json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def test_utils():
    """测试工具函数"""
    print("Testing utility functions...")
    
    # 测试棋盘转换
    board = np.zeros((15, 15), dtype=int)
    board[7, 7] = 1
    board[7, 8] = 2
    
    board_str = board_to_string(board)
    board_reconstructed = string_to_board(board_str)
    
    assert np.array_equal(board, board_reconstructed), "Board conversion failed"
    print("✓ Board conversion test passed")
    
    # 测试数据增强
    augmented = augment_board(board)
    assert len(augmented) == 7, "Data augmentation failed"
    print("✓ Data augmentation test passed")
    
    # 测试复杂度计算
    complexity = calculate_game_complexity(board)
    assert 0 <= complexity <= 1, "Complexity calculation failed"
    print("✓ Complexity calculation test passed")
    
    # 测试位置分析
    analysis = analyze_position(board, 1)
    assert 'threats' in analysis, "Position analysis failed"
    print("✓ Position analysis test passed")
    
    print("All utility function tests passed!")


if __name__ == "__main__":
    test_utils()
