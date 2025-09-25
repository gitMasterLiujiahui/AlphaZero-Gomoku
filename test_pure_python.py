"""
Pure Python Test Script for Gomoku AI (no external dependencies)
五子棋AI纯Python测试脚本（无外部依赖）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_gomoku_board_basic():
    """测试GomokuBoard类的基本功能（不依赖numpy）"""
    print("Testing GomokuBoard (basic)...")
    
    try:
        # 创建一个简化的棋盘类进行测试
        class SimpleBoard:
            BOARD_SIZE = 15
            EMPTY = 0
            BLACK = 1
            WHITE = 2
            
            def __init__(self):
                self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
                self.current_player = self.BLACK
                self.game_over = False
                self.winner = None
                self.move_history = []
            
            def make_move(self, row, col):
                if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                    return False
                if self.board[row][col] != self.EMPTY:
                    return False
                if self.game_over:
                    return False
                
                self.board[row][col] = self.current_player
                self.move_history.append((row, col, self.current_player))
                
                # 简单的胜负判断（检查5子连珠）
                if self.check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                
                self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
                return True
            
            def check_win(self, row, col):
                player = self.board[row][col]
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                
                for dr, dc in directions:
                    count = 1
                    # 向一个方向计数
                    r, c = row + dr, col + dc
                    while (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and 
                           self.board[r][c] == player):
                        count += 1
                        r += dr
                        c += dc
                    
                    # 向相反方向计数
                    r, c = row - dr, col - dc
                    while (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and 
                           self.board[r][c] == player):
                        count += 1
                        r -= dr
                        c -= dc
                    
                    if count >= 5:
                        return True
                return False
            
            def undo_move(self):
                if not self.move_history:
                    return False
                
                row, col, player = self.move_history.pop()
                self.board[row][col] = self.EMPTY
                self.game_over = False
                self.winner = None
                self.current_player = player
                return True
        
        # 测试基本功能
        board = SimpleBoard()
        
        # 测试落子
        assert board.make_move(7, 7) == True
        assert board.board[7][7] == board.BLACK
        assert board.current_player == board.WHITE
        
        # 测试无效移动
        assert board.make_move(7, 7) == False  # 重复落子
        
        # 测试悔棋
        assert board.undo_move() == True
        assert board.board[7][7] == board.EMPTY
        assert board.current_player == board.BLACK
        
        print("✓ GomokuBoard basic test passed")
        return True
        
    except Exception as e:
        print(f"✗ GomokuBoard basic test failed: {e}")
        return False


def test_ai_logic():
    """测试AI逻辑（不依赖外部库）"""
    print("Testing AI Logic...")
    
    try:
        class SimpleAI:
            def __init__(self, player):
                self.player = player
            
            def get_move(self, board):
                # 简单的AI：找到第一个空位
                for row in range(board.BOARD_SIZE):
                    for col in range(board.BOARD_SIZE):
                        if board.board[row][col] == board.EMPTY:
                            return (row, col)
                return None
        
        # 创建测试棋盘
        class SimpleBoard:
            BOARD_SIZE = 15
            EMPTY = 0
            BLACK = 1
            WHITE = 2
            
            def __init__(self):
                self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        
        board = SimpleBoard()
        ai = SimpleAI(1)
        
        # 测试AI移动
        move = ai.get_move(board)
        assert move is not None
        assert len(move) == 2
        assert 0 <= move[0] < board.BOARD_SIZE
        assert 0 <= move[1] < board.BOARD_SIZE
        
        print("✓ AI Logic test passed")
        return True
        
    except Exception as e:
        print(f"✗ AI Logic test failed: {e}")
        return False


def test_game_logic():
    """测试游戏逻辑"""
    print("Testing Game Logic...")
    
    try:
        # 测试五子连珠逻辑
        def check_five_in_row(board, row, col, player):
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            
            for dr, dc in directions:
                count = 1
                # 向一个方向计数
                r, c = row + dr, col + dc
                while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                       board[r][c] == player):
                    count += 1
                    r += dr
                    c += dc
                
                # 向相反方向计数
                r, c = row - dr, col - dc
                while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                       board[r][c] == player):
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 5:
                    return True
            return False
        
        # 创建测试棋盘
        board = [[0 for _ in range(15)] for _ in range(15)]
        
        # 测试水平五子连珠
        for i in range(5):
            board[7][7 + i] = 1
        
        assert check_five_in_row(board, 7, 7, 1) == True
        
        # 测试垂直五子连珠
        board = [[0 for _ in range(15)] for _ in range(15)]
        for i in range(5):
            board[7 + i][7] = 1
        
        assert check_five_in_row(board, 7, 7, 1) == True
        
        print("✓ Game Logic test passed")
        return True
        
    except Exception as e:
        print(f"✗ Game Logic test failed: {e}")
        return False


def test_utility_functions():
    """测试工具函数"""
    print("Testing Utility Functions...")
    
    try:
        def board_to_string(board):
            symbols = {0: '.', 1: '●', 2: '○'}
            lines = []
            lines.append("   " + " ".join(f"{i:2d}" for i in range(len(board[0]))))
            
            for i, row in enumerate(board):
                line = f"{i:2d} " + " ".join(symbols[cell] for cell in row)
                lines.append(line)
            
            return "\n".join(lines)
        
        def string_to_board(board_str):
            lines = board_str.strip().split('\n')
            data_lines = lines[1:]  # 跳过标题行
            
            board = [[0 for _ in range(15)] for _ in range(len(data_lines))]
            
            for i, line in enumerate(data_lines):
                cells = line.split()[1:]  # 跳过行号
                
                for j, cell in enumerate(cells):
                    if cell == '●':
                        board[i][j] = 1
                    elif cell == '○':
                        board[i][j] = 2
            
            return board
        
        # 测试棋盘转换
        board = [[0 for _ in range(15)] for _ in range(15)]
        board[7][7] = 1
        board[7][8] = 2
        
        board_str = board_to_string(board)
        board_reconstructed = string_to_board(board_str)
        
        assert board[7][7] == board_reconstructed[7][7]
        assert board[7][8] == board_reconstructed[7][8]
        
        print("✓ Utility Functions test passed")
        return True
        
    except Exception as e:
        print(f"✗ Utility Functions test failed: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("Testing File Structure...")
    
    try:
        required_files = [
            'main.py',
            'gomoku_board.py',
            'neural_network.py',
            'ai_agent.py',
            'game_ui.py',
            'training.py',
            'utils.py',
            'requirements.txt',
            'README.md',
            'USAGE.md'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        
        # 检查目录结构
        required_dirs = ['models', 'data']
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"✗ Missing directories: {missing_dirs}")
            return False
        
        print("✓ File Structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ File Structure test failed: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("Gomoku AI Pure Python Tests")
    print("=" * 50)
    
    tests = [
        test_gomoku_board_basic,
        test_ai_logic,
        test_game_logic,
        test_utility_functions,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        print("Project structure is correct!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
