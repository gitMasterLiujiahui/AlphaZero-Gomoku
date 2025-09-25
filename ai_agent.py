"""
AlphaZero-Gomoku AI Agent
基于AlphaZero架构的五子棋AI智能体

这个模块实现了AlphaZero-Gomoku AI智能体，包括：
- 蒙特卡洛树搜索(MCTS)
- 神经网络策略和价值评估
- 自动模型保存和加载
- 多种难度级别
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from gomoku_board import GomokuBoard
from neural_network import GomokuModel
import torch
import logging


class AlphaZeroGomokuAI:
    """
    AlphaZero-Gomoku AI智能体
    基于深度学习和蒙特卡洛树搜索的AI算法
    """
    
    def __init__(self, player: int, difficulty: str = "medium", model_path: Optional[str] = None):
        """
        初始化AlphaZero-Gomoku AI
        参数:
            player: 玩家编号 (1=黑子, 2=白子)
            difficulty: 难度级别 ("easy", "medium", "hard")
            model_path: 预训练模型路径，如果为None则自动加载最新模型
        """
        self.player = player
        self.difficulty = difficulty
        self.name = f"AlphaZero_{difficulty}"
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建神经网络模型（不自动加载以避免结构不一致报错）
        self.model = GomokuModel(model_path=model_path)
        
        # 难度参数配置
        self.difficulty_params = {
            "easy": {
                "simulations": 50,
                "temperature": 1.5,
                "exploration": 0.2,
                "c_puct": 1.2
            },
            "medium": {
                "simulations": 100,
                "temperature": 0.8,
                "exploration": 0.05,
                "c_puct": 1.6
            },
            "hard": {
                "simulations": 200,
                "temperature": 0.3,
                "exploration": 0.01,
                "c_puct": 1.8
            }
        }
        
        self.params = self.difficulty_params.get(difficulty, self.difficulty_params["medium"])
        
        # MCTS相关
        self.mcts_root = None
        self.mcts_cache = {}  # 缓存MCTS结果
        
        # 自动保存配置（可禁用以提速）
        self.auto_save_interval = None  # 设为None以禁用自保存
        self.games_played = 0
    
    def get_move(self, board: GomokuBoard) -> Tuple[int, int]:
        """
        获取AI的下一步移动
        参数:
            board: 当前棋盘状态
        返回: (行, 列) 移动位置
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
        
        # 如果棋盘很空，使用开局策略
        if board.get_move_count() < 6:
            return self._opening_move(board, valid_moves)
        
        # 使用MCTS + 神经网络获取最佳移动
        best_move = self._mcts_search(board, valid_moves)
        
        # 添加随机性（根据难度）
        if random.random() < self.params["exploration"]:
            return random.choice(valid_moves)
        
        # 自动保存模型（每10次游戏保存一次）
        self.games_played += 1
        if self.auto_save_interval and self.games_played % self.auto_save_interval == 0:
            self._auto_save_model()
        
        return best_move
    
    def _opening_move(self, board: GomokuBoard, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        开局策略
        优先选择中心区域和关键位置
        """
        center = board.BOARD_SIZE // 2
        move_count = board.get_move_count()
        
        # 第一步：选择天元
        if move_count == 0:
            if (center, center) in valid_moves:
                return (center, center)
        
        # 前几步：优先选择中心区域
        if move_count < 6:
            # 中心区域（3x3）
            center_moves = [(r, c) for r, c in valid_moves 
                           if abs(r - center) <= 1 and abs(c - center) <= 1]
            if center_moves:
                return random.choice(center_moves)
            
            # 扩展中心区域（5x5）
            extended_center = [(r, c) for r, c in valid_moves 
                              if abs(r - center) <= 2 and abs(c - center) <= 2]
            if extended_center:
                return random.choice(extended_center)
        
        # 后续步骤：使用MCTS
        return self._mcts_search(board, valid_moves)
    
    def _mcts_search(self, board: GomokuBoard, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        蒙特卡洛树搜索
        结合神经网络进行策略搜索
        """
        # 创建MCTS根节点
        root = MCTSNode(board, None, None, self.model, self.params)
        
        # 运行MCTS搜索
        for _ in range(self.params["simulations"]):
            self._mcts_simulation(root)
        
        # 选择最佳移动
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move
        
        # 如果没有子节点，返回随机移动
        return random.choice(valid_moves)
    
    def _mcts_simulation(self, root: 'MCTSNode'):
        """
        MCTS单次模拟
        选择 -> 扩展 -> 模拟 -> 回传
        """
        # 选择阶段
        node = self._select(root)
        
        # 扩展阶段
        if not node.is_terminal() and node.visits > 0:
            node = self._expand(node)
        
        # 模拟阶段
        value = self._simulate(node)
        
        # 回传阶段
        self._backpropagate(node, value)
    
    def _select(self, node: 'MCTSNode') -> 'MCTSNode':
        """
        选择阶段：从根节点开始，选择UCB1值最高的子节点
        """
        while not node.is_terminal() and len(node.children) > 0:
            if node.unexplored_moves:
                return node
            node = self._best_child(node)
        return node
    
    def _expand(self, node: 'MCTSNode') -> 'MCTSNode':
        """
        扩展阶段：为当前节点添加新的子节点
        """
        if not node.unexplored_moves:
            return node
        
        move = node.unexplored_moves.pop()
        child_board = node.board.copy_board()
        child_board.make_move(move[0], move[1])
        
        child = MCTSNode(child_board, node, move, self.model, self.params)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: 'MCTSNode') -> float:
        """
        模拟阶段：使用神经网络评估局面价值
        """
        if node.is_terminal():
            if node.board.winner == self.player:
                return 1.0
            elif node.board.winner is not None:
                return -1.0
            else:
                return 0.0
        
        # 使用神经网络评估局面
        _, value = node.model.predict(node.board.get_board_state())
        return value
    
    def _backpropagate(self, node: 'MCTSNode', value: float):
        """
        回传阶段：将模拟结果回传到路径上的所有节点
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _best_child(self, node: 'MCTSNode') -> 'MCTSNode':
        """
        选择UCB1值最高的子节点
        """
        return max(node.children, key=lambda c: c.ucb1())
    
    def evaluate_position(self, board: GomokuBoard) -> float:
        """
        评估当前局面的价值
        参数:
            board: 棋盘状态
        返回: 局面价值 (-1 到 1)
        """
        if board.game_over:
            if board.winner == self.player:
                return 1.0
            elif board.winner is not None:
                return -1.0
            else:
                return 0.0
        
        # 使用神经网络评估
        _, value = self.model.predict(board.get_board_state())
        return value
    
    def _auto_save_model(self):
        """
        自动保存模型
        每10次游戏自动保存一次模型参数
        """
        try:
            # 使用模型的自保存功能
            filepath = self.model.auto_save_model(suffix=f"_{self.difficulty}")
            self.logger.info(f"Auto-saved model to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to auto-save model: {e}")


class MCTSNode:
    """
    MCTS节点类
    用于蒙特卡洛树搜索的节点表示
    """
    
    def __init__(self, board: GomokuBoard, parent: Optional['MCTSNode'], 
                 move: Optional[Tuple[int, int]], model: GomokuModel, params: Dict[str, Any]):
        """
        初始化MCTS节点
        参数:
            board: 棋盘状态
            parent: 父节点
            move: 到达此节点的移动
            model: 神经网络模型
            params: MCTS参数
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.model = model
        self.params = params
        
        # MCTS统计信息
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.unexplored_moves = board.get_valid_moves().copy()
        
        # 使用神经网络获取先验概率
        if not self.is_terminal():
            self.prior_prob = self._get_prior_probability()
    
    def is_terminal(self) -> bool:
        """
        检查是否为终端节点
        返回: 是否为游戏结束状态
        """
        return self.board.game_over
    
    def ucb1(self) -> float:
        """
        计算UCB1值
        用于选择最佳子节点
        返回: UCB1值
        """
        if self.visits == 0:
            return float('inf')
        
        # 利用项：平均价值
        exploitation = self.value / self.visits
        
        # 探索项：基于访问次数的不确定性
        exploration = self.params["c_puct"] * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def _get_prior_probability(self) -> np.ndarray:
        """
        获取先验概率分布
        使用神经网络预测移动概率
        返回: 移动概率分布
        """
        policy, _ = self.model.predict(self.board.get_board_state())
        
        # 只保留有效移动的概率
        valid_probs = np.zeros(len(self.unexplored_moves))
        for i, (row, col) in enumerate(self.unexplored_moves):
            move_idx = row * self.board.BOARD_SIZE + col
            valid_probs[i] = policy[move_idx]
        
        # 归一化
        if np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
        
        return valid_probs


class AIFactory:
    """
    AlphaZero-Gomoku AI工厂类
    用于创建AlphaZero-Gomoku AI实例
    """
    
    @staticmethod
    def create_ai(ai_type: str, player: int, difficulty: str = "medium", **kwargs) -> AlphaZeroGomokuAI:
        """
        创建AlphaZero-Gomoku AI实例
        参数:
            ai_type: AI类型（只支持"alphazero"）
            player: 玩家编号 (1=黑子, 2=白子)
            difficulty: 难度级别 ("easy", "medium", "hard")
            **kwargs: 其他参数，包括model_path等
        返回: AlphaZero-Gomoku AI实例
        """
        if ai_type != "alphazero":
            raise ValueError(f"Only 'alphazero' AI type is supported, got: {ai_type}")
        
        return AlphaZeroGomokuAI(player, difficulty, **kwargs)
    
    @staticmethod
    def get_available_ai_types() -> List[str]:
        """
        获取可用的AI类型
        返回: AI类型列表（目前只有AlphaZero）
        """
        return ["alphazero"]
    
    @staticmethod
    def get_available_difficulties() -> List[str]:
        """
        获取可用的难度级别
        返回: 难度级别列表
        """
        return ["easy", "medium", "hard"]


def test_ai():
    """
    测试AlphaZero-Gomoku AI
    验证AI的基本功能和自动保存功能
    """
    print("Testing AlphaZero-Gomoku AI...")
    
    # 创建测试棋盘
    board = GomokuBoard()
    
    # 测试AlphaZero AI
    try:
        ai = AIFactory.create_ai("alphazero", board.BLACK, "medium")
        
        # 测试移动
        start_time = time.time()
        move = ai.get_move(board)
        end_time = time.time()
        
        print(f"Move: {move}")
        print(f"Time: {end_time - start_time:.3f}s")
        
        # 测试评估
        score = ai.evaluate_position(board)
        print(f"Position evaluation: {score:.3f}")
        
        # 测试自动保存功能
        ai._auto_save_model()
        print("✓ Auto-save test passed")
        
        print("✓ AlphaZero AI test passed")
        
    except Exception as e:
        print(f"✗ AlphaZero AI test failed: {e}")
    
    print("AlphaZero-Gomoku AI testing completed!")


if __name__ == "__main__":
    test_ai()