"""
AlphaZero-Gomoku AI Game Main Program
基于AlphaZero架构的五子棋AI游戏主程序

这个模块是五子棋AI游戏的主入口，提供：
- 游戏主循环管理
- 性能监控
- 配置管理
- 依赖检查
- 组件测试
"""

import sys
import os
import logging
import traceback
import time
from typing import Optional, Dict, Any
import pygame

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gomoku_board import GomokuBoard
from ai_agent import AIFactory, AlphaZeroGomokuAI
from neural_network import GomokuModel
from game_ui import GomokuUI


class GomokuGame:
    """
    AlphaZero-Gomoku游戏主类
    管理游戏的整体流程和状态
    """
    
    def __init__(self):
        """
        初始化游戏实例
        设置日志、UI和游戏统计
        """
        self.setup_logging()
        self.ui = None
        self.game_stats = {
            'games_played': 0,    # 已玩游戏数
            'human_wins': 0,      # 人类获胜次数
            'ai_wins': 0,         # AI获胜次数
            'draws': 0,           # 平局次数
            'total_moves': 0      # 总移动次数
        }
        
    def setup_logging(self):
        """
        设置日志系统
        配置日志格式和输出方式
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gomoku.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_ui(self) -> bool:
        """
        初始化游戏UI
        返回: 是否初始化成功
        """
        try:
            self.ui = GomokuUI()
            self.ui.set_callbacks(self.on_move_made, self.on_game_over)
            self.logger.info("UI initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize UI: {e}")
            return False
    
    def on_move_made(self, move: tuple, player: int):
        """
        移动回调函数
        参数:
            move: 移动位置 (行, 列)
            player: 玩家编号
        """
        self.game_stats['total_moves'] += 1
        self.logger.info(f"Move made: {move} by player {player}")
    
    def on_game_over(self, winner: Optional[int]):
        """
        游戏结束回调函数
        参数:
            winner: 获胜者编号 (1=黑子, 2=白子, None=平局)
        """
        self.game_stats['games_played'] += 1
        
        if winner == GomokuBoard.BLACK:
            self.game_stats['human_wins'] += 1
            self.logger.info("Game over: Human (Black) wins!")
        elif winner == GomokuBoard.WHITE:
            self.game_stats['ai_wins'] += 1
            self.logger.info("Game over: AI (White) wins!")
        else:
            self.game_stats['draws'] += 1
            self.logger.info("Game over: Draw!")
        
        # 显示游戏统计
        self.show_game_stats()
    
    def show_game_stats(self):
        """
        显示游戏统计信息
        包括胜负记录和胜率统计
        """
        stats = self.game_stats
        self.logger.info(f"Game Statistics:")
        self.logger.info(f"  Games played: {stats['games_played']}")
        self.logger.info(f"  Human wins: {stats['human_wins']}")
        self.logger.info(f"  AI wins: {stats['ai_wins']}")
        self.logger.info(f"  Draws: {stats['draws']}")
        self.logger.info(f"  Total moves: {stats['total_moves']}")
        
        if stats['games_played'] > 0:
            win_rate = stats['human_wins'] / stats['games_played'] * 100
            self.logger.info(f"  Human win rate: {win_rate:.1f}%")
    
    def run(self):
        """
        运行游戏主循环
        返回: 是否运行成功
        """
        try:
            self.logger.info("Starting AlphaZero-Gomoku AI Game...")
            
            # 初始化UI
            if not self.initialize_ui():
                self.logger.error("Failed to initialize UI")
                return False
            
            # 运行游戏循环
            self.ui.run()
            
        except KeyboardInterrupt:
            self.logger.info("Game interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """
        清理游戏资源
        释放UI和pygame资源
        """
        self.logger.info("Cleaning up resources...")
        if self.ui:
            pygame.quit()
        self.logger.info("Game ended")


class PerformanceMonitor:
    """
    性能监控器
    监控游戏和AI的性能指标
    """
    
    def __init__(self):
        """
        初始化性能监控器
        记录开始时间和各种时间统计
        """
        self.start_time = time.time()
        self.move_times = []  # 移动时间记录
        self.ai_times = []    # AI计算时间记录
    
    def start_move(self):
        """开始移动计时"""
        self.move_start = time.time()
    
    def end_move(self):
        """结束移动计时并记录时间"""
        move_time = time.time() - self.move_start
        self.move_times.append(move_time)
    
    def start_ai(self):
        """开始AI计算计时"""
        self.ai_start = time.time()
    
    def end_ai(self):
        """结束AI计算计时并记录时间"""
        ai_time = time.time() - self.ai_start
        self.ai_times.append(ai_time)
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        返回: 包含各种性能指标的字典
        """
        total_time = time.time() - self.start_time
        
        stats = {
            'total_time': total_time,           # 总运行时间
            'total_moves': len(self.move_times), # 总移动次数
            'total_ai_calls': len(self.ai_times) # 总AI调用次数
        }
        
        if self.move_times:
            stats['avg_move_time'] = sum(self.move_times) / len(self.move_times)
            stats['max_move_time'] = max(self.move_times)
        
        if self.ai_times:
            stats['avg_ai_time'] = sum(self.ai_times) / len(self.ai_times)
            stats['max_ai_time'] = max(self.ai_times)
        
        return stats


class GameConfig:
    """
    游戏配置类
    管理游戏的各项配置参数
    """
    
    def __init__(self):
        """
        初始化游戏配置
        设置默认配置参数
        """
        self.board_size = 15                    # 棋盘大小
        self.default_ai_type = "alphazero"      # 默认AI类型（只支持AlphaZero）
        self.default_difficulty = "medium"      # 默认难度
        self.enable_animations = True           # 启用动画
        self.enable_sound = False               # 启用声音
        self.auto_save = True                   # 自动保存
        self.max_undo_moves = 10                # 最大悔棋次数
    
    def load_from_file(self, filename: str = "config.json"):
        """
        从文件加载配置
        参数:
            filename: 配置文件路径
        """
        try:
            import json
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
    
    def save_to_file(self, filename: str = "config.json"):
        """
        保存配置到文件
        参数:
            filename: 配置文件路径
        """
        try:
            import json
            config_data = {
                'board_size': self.board_size,
                'default_ai_type': self.default_ai_type,
                'default_difficulty': self.default_difficulty,
                'enable_animations': self.enable_animations,
                'enable_sound': self.enable_sound,
                'auto_save': self.auto_save,
                'max_undo_moves': self.max_undo_moves
            }
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save config: {e}")


def check_dependencies():
    """
    检查项目依赖包
    返回: 是否所有依赖包都已安装
    """
    required_packages = ['pygame', 'torch', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """
    主函数
    程序入口点，负责初始化和运行游戏
    返回: 退出代码
    """
    print("=" * 50)
    print("AlphaZero-Gomoku AI 游戏")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 创建游戏实例
    game = GomokuGame()
    
    # 加载配置
    config = GameConfig()
    config.load_from_file()
    
    # 运行游戏
    try:
        success = game.run()
        return 0 if success else 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


def test_components():
    """
    测试各个组件
    验证游戏各个模块是否正常工作
    """
    print("Testing AlphaZero-Gomoku components...")
    
    # 测试棋盘
    print("Testing GomokuBoard...")
    try:
        from gomoku_board import GomokuBoard
        board = GomokuBoard()
        board.make_move(7, 7)
        print("✓ GomokuBoard test passed")
    except Exception as e:
        print(f"✗ GomokuBoard test failed: {e}")
    
    # 测试AlphaZero AI
    print("Testing AlphaZero AI Agent...")
    try:
        from ai_agent import AIFactory
        ai = AIFactory.create_ai("alphazero", 1)
        move = ai.get_move(board)
        print("✓ AlphaZero AI Agent test passed")
    except Exception as e:
        print(f"✗ AlphaZero AI Agent test failed: {e}")
    
    # 测试神经网络
    print("Testing AlphaZero Neural Network...")
    try:
        from neural_network import GomokuModel
        model = GomokuModel()
        policy, value = model.predict(board.get_board_state())
        print("✓ AlphaZero Neural Network test passed")
    except Exception as e:
        print(f"✗ AlphaZero Neural Network test failed: {e}")
    
    print("AlphaZero-Gomoku component testing completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_components()
    else:
        exit_code = main()
        sys.exit(exit_code)
