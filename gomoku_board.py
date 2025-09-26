"""
Gomoku Board Implementation
五子棋棋盘逻辑实现

这个模块实现了五子棋游戏的核心逻辑，包括：
- 15x15棋盘表示
- 落子、悔棋、胜负判断
- 棋盘状态复制和比较
- 游戏状态管理
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import copy


class GomokuBoard:
    """
    五子棋棋盘类
    
    实现了五子棋游戏的核心逻辑，包括棋盘状态管理、
    移动验证、胜负判断等功能。
    """
    
    # 棋盘常量定义
    BOARD_SIZE = 15          # 棋盘大小：15x15
    EMPTY = 0                # 空位
    BLACK = 1                # 黑子（先手）
    WHITE = 2                # 白子（后手）
    
    # 方向向量定义（用于检查五子连珠）
    # 格式：(行变化, 列变化)
    DIRECTIONS = [
        (0, 1),   # 水平方向
        (1, 0),   # 垂直方向
        (1, 1),   # 主对角线方向
        (1, -1)   # 副对角线方向
    ]
    
    def __init__(self):
        """
        初始化五子棋棋盘
        
        创建15x15的空棋盘，初始化游戏状态
        """
        # 初始化棋盘为全空状态
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        
        # 游戏状态管理
        self.move_history = []           # 落子历史记录
        self.current_player = self.BLACK  # 当前玩家（黑子先行）
        self.game_over = False            # 游戏是否结束
        self.winner = None                # 获胜者（None表示平局）
        
    def reset(self):
        """
        重置棋盘到初始状态
        
        清空所有棋子，重置游戏状态，准备新游戏
        """
        self.board.fill(self.EMPTY)        # 清空棋盘
        self.move_history.clear()          # 清空历史记录
        self.current_player = self.BLACK   # 重置为黑子先行
        self.game_over = False             # 重置游戏状态
        self.winner = None                 # 清空获胜者
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        检查落子位置是否有效
        
        参数:
            row: 行坐标 (0-14)
            col: 列坐标 (0-14)
        返回:
            bool: 位置是否有效
        """
        # 检查坐标是否在棋盘范围内
        if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
            return False
        
        # 检查位置是否为空
        return self.board[row, col] == self.EMPTY
    
    def make_move(self, row: int, col: int) -> bool:
        """
        在指定位置落子
        
        参数:
            row: 行坐标
            col: 列坐标
        返回:
            bool: 落子是否成功
        """
        # 验证移动有效性
        if not self.is_valid_move(row, col) or self.game_over:
            return False
        
        # 记录落子信息
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # 检查游戏是否结束
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full() or self.get_move_count() >= 200:  # 增加步数限制避免过长对局
            self.game_over = True
            self.winner = None  # 平局
        
        # 切换玩家
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
        
        return True
    
    def undo_move(self) -> bool:
        """
        悔棋操作
        
        撤销最后一步落子，恢复到上一步状态
        
        返回:
            bool: 悔棋是否成功
        """
        # 检查是否有历史记录
        if not self.move_history:
            return False
        
        # 移除最后一步
        row, col, player = self.move_history.pop()
        self.board[row, col] = self.EMPTY
        
        # 重置游戏状态
        self.game_over = False
        self.winner = None
        self.current_player = player  # 恢复到上一步的玩家
        
        return True
    
    def check_win(self, row: int, col: int) -> bool:
        """
        检查是否获胜（五子连珠）
        
        参数:
            row: 最后落子的行坐标
            col: 最后落子的列坐标
        返回:
            bool: 是否获胜
        """
        player = self.board[row, col]
        
        # 检查四个方向是否有五子连珠
        for direction in self.DIRECTIONS:
            count = 1  # 包含当前棋子
            
            # 向一个方向计数连续棋子
            count += self._count_consecutive(row, col, direction, player)
            
            # 向相反方向计数连续棋子
            count += self._count_consecutive(row, col, (-direction[0], -direction[1]), player)
            
            # 如果连续棋子数达到5个，则获胜
            if count >= 5:
                return True
        
        return False
    
    def _count_consecutive(self, row: int, col: int, direction: Tuple[int, int], player: int) -> int:
        """
        计算指定方向上的连续同色棋子数量
        
        参数:
            row: 起始行坐标
            col: 起始列坐标
            direction: 方向向量 (行变化, 列变化)
            player: 玩家编号
        返回:
            int: 连续棋子数量
        """
        count = 0
        dr, dc = direction
        
        # 向指定方向移动并计数
        r, c = row + dr, col + dc
        while (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and 
               self.board[r, c] == player):
            count += 1
            r += dr
            c += dc
        
        return count
    
    def is_board_full(self) -> bool:
        """
        检查棋盘是否已满
        
        返回:
            bool: 棋盘是否已满
        """
        return np.all(self.board != self.EMPTY)
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        获取所有有效落子位置
        
        返回:
            List[Tuple[int, int]]: 有效位置列表
        """
        moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.is_valid_move(row, col):
                    moves.append((row, col))
        return moves
    
    def get_board_state(self) -> np.ndarray:
        """
        获取当前棋盘状态
        
        返回:
            np.ndarray: 棋盘状态副本
        """
        return self.board.copy()
    
    def copy_board(self) -> 'GomokuBoard':
        """
        复制当前棋盘状态
        
        返回:
            GomokuBoard: 棋盘副本
        """
        new_board = GomokuBoard()
        new_board.board = self.board.copy()
        new_board.move_history = self.move_history.copy()
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        return new_board
    
    def get_board_tensor(self) -> np.ndarray:
        """
        获取棋盘张量表示（用于神经网络）
        
        将棋盘转换为3通道张量：
        - 通道0：黑子位置
        - 通道1：白子位置  
        - 通道2：空位位置
        
        返回:
            np.ndarray: 3通道张量 (3, 15, 15)
        """
        tensor = np.zeros((3, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        
        # 黑子通道
        tensor[0] = (self.board == self.BLACK).astype(np.float32)
        # 白子通道
        tensor[1] = (self.board == self.WHITE).astype(np.float32)
        # 空位通道
        tensor[2] = (self.board == self.EMPTY).astype(np.float32)
        
        return tensor
    
    def get_last_move(self) -> Optional[Tuple[int, int]]:
        """
        获取最后一步落子位置
        
        返回:
            Optional[Tuple[int, int]]: 最后一步的位置，如果没有则返回None
        """
        if not self.move_history:
            return None
        return self.move_history[-1][:2]
    
    def get_move_count(self) -> int:
        """
        获取总步数
        
        返回:
            int: 已下棋子总数
        """
        return len(self.move_history)
    
    def get_game_status(self) -> dict:
        """
        获取游戏状态信息
        
        返回:
            dict: 包含游戏状态的字典
        """
        return {
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'move_count': self.get_move_count(),
            'board_full': self.is_board_full()
        }
    
    def __str__(self) -> str:
        """
        棋盘字符串表示
        
        返回:
            str: 格式化的棋盘字符串
        """
        symbols = {self.EMPTY: '.', self.BLACK: '●', self.WHITE: '○'}
        
        lines = []
        lines.append("   " + " ".join(f"{i:2d}" for i in range(self.BOARD_SIZE)))
        
        for i, row in enumerate(self.board):
            line = f"{i:2d} " + " ".join(symbols[cell] for cell in row)
            lines.append(line)
        
        return "\n".join(lines)
    
    def __eq__(self, other) -> bool:
        """
        比较两个棋盘是否相等
        
        参数:
            other: 另一个棋盘对象
        返回:
            bool: 是否相等
        """
        if not isinstance(other, GomokuBoard):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)


def test_gomoku_board():
    """
    测试GomokuBoard类的基本功能
    
    验证棋盘的各种操作是否正常工作
    """
    print("Testing GomokuBoard class...")
    
    # 创建棋盘实例
    board = GomokuBoard()
    
    # 测试基本功能
    print("Initial board:")
    print(board)
    print(f"Current player: {board.current_player}")
    print(f"Valid moves count: {len(board.get_valid_moves())}")
    
    # 测试落子功能
    print("\nTesting move making:")
    success = board.make_move(7, 7)  # 中心落子
    print(f"Move (7,7) successful: {success}")
    print(board)
    
    # 测试获胜条件
    print("\nTesting win condition:")
    test_board = GomokuBoard()
    # 创建五子连珠
    for i in range(5):
        test_board.make_move(7, 7 + i)
    print(test_board)
    print(f"Game over: {test_board.game_over}")
    print(f"Winner: {test_board.winner}")
    
    # 测试悔棋功能
    print("\nTesting undo move:")
    board.make_move(6, 6)
    print("Before undo:")
    print(board)
    board.undo_move()
    print("After undo:")
    print(board)
    
    print("GomokuBoard test completed!")


if __name__ == "__main__":
    test_gomoku_board()
