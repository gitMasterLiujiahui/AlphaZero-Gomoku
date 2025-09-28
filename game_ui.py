"""
AlphaZero-Gomoku Game Pygame UI
基于AlphaZero架构的五子棋游戏Pygame界面

这个模块实现了五子棋游戏的图形用户界面，包括：
- 游戏主菜单和设置界面
- 棋盘绘制和交互
- 游戏状态显示
- 动画效果
- 键盘快捷键支持
"""

import pygame
import sys
import time
from typing import Tuple, Optional, Callable, Dict, Any
from gomoku_board import GomokuBoard
from ai_agent import AIFactory, AlphaZeroGomokuAI


# 颜色定义
class Colors:
    """
    游戏界面颜色定义
    包含所有UI元素的颜色配置
    """
    # 基础颜色
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    BROWN = (139, 69, 19)
    LIGHT_BROWN = (222, 184, 135)
    
    # 棋子颜色
    BLACK_STONE = (45, 45, 45)        # 黑子颜色
    WHITE_STONE = (245, 245, 245)     # 白子颜色
    BLACK_STONE_BORDER = (20, 20, 20) # 黑子边框
    WHITE_STONE_BORDER = (200, 200, 200) # 白子边框


class GomokuUI:
    """
    AlphaZero-Gomoku游戏界面
    管理游戏的图形用户界面和交互
    """
    
    def __init__(self, board_size: int = 15):
        """
        初始化游戏界面
        参数:
            board_size: 棋盘大小，默认15x15
        """
        pygame.init()
        
        # 界面尺寸配置
        self.board_size = board_size
        self.cell_size = 40          # 每个格子的像素大小
        self.margin = 50             # 边距
        self.stone_radius = 18       # 棋子半径
        
        # 计算窗口大小
        self.board_width = self.board_size * self.cell_size
        self.board_height = self.board_size * self.cell_size
        self.ui_width = 300          # UI面板宽度
        self.window_width = self.board_width + 2 * self.margin + self.ui_width
        self.window_height = self.board_height + 2 * self.margin
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("AlphaZero-Gomoku AI")
        
        # 字体配置 - 使用系统默认字体以确保中文显示
        try:
            # 尝试使用系统中文字体
            self.font_large = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
            self.font_medium = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 24)
            self.font_small = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 18)
        except:
            try:
                # 备用字体
                self.font_large = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 36)
                self.font_medium = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 24)
                self.font_small = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 18)
            except:
                # 最后使用默认字体
                self.font_large = pygame.font.Font(None, 36)
                self.font_medium = pygame.font.Font(None, 24)
                self.font_small = pygame.font.Font(None, 18)
        
        # 游戏状态
        self.board = GomokuBoard()
        self.ai_player = None
        self.human_player = GomokuBoard.BLACK
        self.game_mode = "human_vs_ai"  # 游戏模式
        self.difficulty = "medium"      # AI难度
        self.ai_type = "alphazero"      # AI类型（alphazero 或 bg_planner）
        
        # UI状态
        self.selected_difficulty = "medium"
        self.selected_ai_type = "alphazero"
        self.selected_color = "black"   # 人类玩家选择的颜色
        self.show_menu = True
        self.show_settings = False
        self.game_started = False
        
        # 动画效果
        self.last_move_pos = None
        self.animation_time = 0
        self.animation_duration = 0.3
        
        # 回调函数
        self.on_move_made = None
        self.on_game_over = None
        
    def set_callbacks(self, on_move_made: Optional[Callable] = None, 
                     on_game_over: Optional[Callable] = None):
        """
        设置回调函数
        参数:
            on_move_made: 移动回调函数
            on_game_over: 游戏结束回调函数
        """
        self.on_move_made = on_move_made
        self.on_game_over = on_game_over
    
    def start_game(self, game_mode: str = "human_vs_ai", difficulty: str = "medium"):
        """
        开始游戏
        参数:
            game_mode: 游戏模式 ("human_vs_ai", "ai_vs_ai", "human_vs_human")
            difficulty: AI难度 ("easy", "medium", "hard")
        """
        self.game_mode = game_mode
        self.difficulty = difficulty
        
        # 根据用户选择设置人类玩家颜色
        if self.selected_color == "black":
            self.human_player = GomokuBoard.BLACK
        else:
            self.human_player = GomokuBoard.WHITE
        
        # 重置棋盘
        self.board.reset()
        
        # 创建AlphaZero AI
        if game_mode in ["human_vs_ai", "ai_vs_ai"]:
            # AI使用与人类相反的颜色
            ai_color = GomokuBoard.WHITE if self.human_player == GomokuBoard.BLACK else GomokuBoard.BLACK
            self.ai_player = AIFactory.create_ai(self.ai_type, ai_color, difficulty, device="cpu")
        
        self.show_menu = False
        self.show_settings = False
        self.game_started = True
        
        # 如果AI先手，让AI下第一步
        if game_mode == "ai_vs_ai" or (game_mode == "human_vs_ai" and self.ai_player.player == GomokuBoard.BLACK):
            self._make_ai_move()
    
    def _make_ai_move(self):
        """
        AI下棋
        让AlphaZero AI计算并执行下一步移动
        """
        if not self.ai_player or self.board.game_over:
            return
        
        # 检查是否轮到AI
        if self.board.current_player != self.ai_player.player:
            return
        
        # 获取AI移动
        move = self.ai_player.get_move(self.board)
        if move:
            self.board.make_move(move[0], move[1])
            self.last_move_pos = move
            self.animation_time = time.time()
            
            # 调用回调
            if self.on_move_made:
                self.on_move_made(move, self.ai_player.player)
            
            # 检查游戏结束
            if self.board.game_over and self.on_game_over:
                # 将对局数据保存为可用于训练的格式
                try:
                    game = {
                        'moves': [(self.board.get_board_state().tolist(), self.last_move_pos, self.board.current_player)],
                        'winner': self.board.winner,
                        'game_length': self.board.get_move_count()
                    }
                    import os, json, time as _t
                    os.makedirs('data/human_games', exist_ok=True)
                    with open(os.path.join('data/human_games', f'game_{int(_t.time())}.json'), 'w', encoding='utf-8') as f:
                        json.dump(game, f, ensure_ascii=False)
                except Exception:
                    pass
                self.on_game_over(self.board.winner)
    
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """处理鼠标点击"""
        if not self.game_started:
            return self._handle_menu_click(pos)
        
        # 如果游戏已结束，点击返回主菜单
        if self.board.game_over:
            self.show_menu = True
            self.show_settings = False
            self.game_started = False
            return True
        
        # 检查是否轮到人类玩家
        if self.game_mode == "human_vs_human" or (self.game_mode == "human_vs_ai" and 
                                                 self.board.current_player == self.human_player):
            return self._handle_board_click(pos)
        
        return False
    
    def _handle_menu_click(self, pos: Tuple[int, int]) -> bool:
        """处理菜单点击"""
        x, y = pos
        
        if self.show_menu:
            # 主菜单
            menu_y = 200
            button_height = 50
            button_spacing = 60
            
            buttons = [
                ("人机对战", "human_vs_ai"),
                ("AI对战", "ai_vs_ai"),
                ("人人对战", "human_vs_human"),
                ("设置", "settings"),
                ("退出", "quit")
            ]
            
            for i, (text, action) in enumerate(buttons):
                button_y = menu_y + i * button_spacing
                if 100 <= x <= 400 and button_y <= y <= button_y + button_height:
                    if action == "settings":
                        self.show_settings = True
                        self.show_menu = False
                    elif action == "quit":
                        return False
                    else:
                        self.start_game(action, self.selected_difficulty)
                    return True
        
        elif self.show_settings:
            # 设置菜单
            if 50 <= x <= 200 and 50 <= y <= 80:
                self.show_settings = False
                self.show_menu = True
                return True
            
            # 难度选择
            difficulties = ["easy", "medium", "hard"]
            for i, difficulty in enumerate(difficulties):
                button_y = 230 + i * 40
                if 100 <= x <= 300 and button_y <= y <= button_y + 30:
                    self.selected_difficulty = difficulty
                    return True

            # AI 类型切换点击
            types = ["alphazero", "bg_planner"]
            for i, t in enumerate(types):
                ty = 190 + i * 30
                if 180 <= x <= 340 and ty <= y <= ty + 24:
                    self.selected_ai_type = t
                    self.ai_type = t
                    return True
            
            # 颜色选择点击
            colors = ["black", "white"]
            for i, color in enumerate(colors):
                color_y = 350 + i * 40
                if 100 <= x <= 300 and color_y <= y <= color_y + 30:
                    self.selected_color = color
                    return True
        
        return False
    
    def _handle_board_click(self, pos: Tuple[int, int]) -> bool:
        """处理棋盘点击"""
        x, y = pos
        
        # 计算棋盘坐标
        board_x = x - self.margin
        board_y = y - self.margin
        
        if not (0 <= board_x <= self.board_width and 0 <= board_y <= self.board_height):
            return False
        
        # 计算格子坐标
        col = round(board_x / self.cell_size)
        row = round(board_y / self.cell_size)
        
        # 检查是否有效移动
        if self.board.is_valid_move(row, col):
            self.board.make_move(row, col)
            self.last_move_pos = (row, col)
            self.animation_time = time.time()
            
            # 调用回调
            if self.on_move_made:
                self.on_move_made((row, col), self.board.current_player)
            
            # 检查游戏结束
            if self.board.game_over and self.on_game_over:
                self.on_game_over(self.board.winner)
            
            # 如果是对战AI，让AI下下一步
            if self.game_mode == "human_vs_ai" and not self.board.game_over:
                pygame.time.wait(500)  # 短暂延迟
                self._make_ai_move()
            
            return True
        
        return False
    
    def handle_key(self, key: int) -> bool:
        """处理键盘输入"""
        if key == pygame.K_r:
            # 重新开始
            self.start_game(self.game_mode, self.ai_type, self.difficulty)
            return True
        elif key == pygame.K_u:
            # 悔棋
            if self.board.undo_move():
                if self.game_mode == "human_vs_ai" and not self.board.game_over:
                    self.board.undo_move()  # AI也悔棋
                return True
        elif key == pygame.K_m:
            # 返回主菜单
            self.show_menu = True
            self.show_settings = False
            self.game_started = False
            return True
        elif key == pygame.K_s:
            # 显示设置
            self.show_settings = True
            self.show_menu = False
            return True
        
        return False
    
    def draw(self):
        """绘制界面"""
        self.screen.fill(Colors.LIGHT_BROWN)
        
        if self.show_menu:
            self._draw_main_menu()
        elif self.show_settings:
            self._draw_settings_menu()
        else:
            self._draw_game()
    
    def _draw_main_menu(self):
        """绘制主菜单"""
        # 标题
        title_text = self.font_large.render("五子棋 AI", True, Colors.BLACK)
        title_rect = title_text.get_rect(center=(self.window_width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # 菜单按钮
        menu_y = 200
        button_height = 50
        button_spacing = 60
        
        buttons = [
            ("人机对战", Colors.BLUE),
            ("AI对战", Colors.GREEN),
            ("人人对战", Colors.RED),
            ("设置", Colors.GRAY),
            ("退出", Colors.DARK_GRAY)
        ]
        
        for i, (text, color) in enumerate(buttons):
            button_y = menu_y + i * button_spacing
            button_rect = pygame.Rect(100, button_y, 300, button_height)
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, Colors.BLACK, button_rect, 2)
            
            text_surface = self.font_medium.render(text, True, Colors.WHITE)
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)
    
    def _draw_settings_menu(self):
        """绘制设置菜单"""
        # 返回按钮
        back_rect = pygame.Rect(50, 50, 150, 30)
        pygame.draw.rect(self.screen, Colors.GRAY, back_rect)
        pygame.draw.rect(self.screen, Colors.BLACK, back_rect, 2)
        
        back_text = self.font_small.render("返回主菜单", True, Colors.WHITE)
        back_text_rect = back_text.get_rect(center=back_rect.center)
        self.screen.blit(back_text, back_text_rect)
        
        # AI类型选择（只显示AlphaZero）
        ai_title = self.font_medium.render(f"AI类型: {self.ai_type}", True, Colors.BLACK)
        self.screen.blit(ai_title, (100, 120))
        
        # 难度选择
        diff_title = self.font_medium.render("难度级别:", True, Colors.BLACK)
        self.screen.blit(diff_title, (100, 200))
        
        difficulties = ["easy", "medium", "hard"]
        for i, difficulty in enumerate(difficulties):
            button_y = 230 + i * 40
            button_rect = pygame.Rect(100, button_y, 200, 30)
            
            color = Colors.GREEN if difficulty == self.selected_difficulty else Colors.LIGHT_GRAY
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, Colors.BLACK, button_rect, 2)
            
            text = self.font_small.render(difficulty, True, Colors.BLACK)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)

        # AI 类型切换按钮
        types = ["alphazero", "bg_planner"]
        self.screen.blit(self.font_medium.render("AI类型:", True, Colors.BLACK), (100, 160))
        for i, t in enumerate(types):
            y = 190 + i * 30
            rect = pygame.Rect(180, y, 160, 24)
            color = Colors.GREEN if t == self.selected_ai_type else Colors.LIGHT_GRAY
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, Colors.BLACK, rect, 2)
            txt = self.font_small.render(t, True, Colors.BLACK)
            self.screen.blit(txt, txt.get_rect(center=rect.center))
        
        # 颜色选择
        color_title = self.font_medium.render("执子颜色:", True, Colors.BLACK)
        self.screen.blit(color_title, (100, 320))
        
        colors = ["black", "white"]
        color_names = ["黑子(先手)", "白子(后手)"]
        for i, (color, name) in enumerate(zip(colors, color_names)):
            color_y = 350 + i * 40
            button_rect = pygame.Rect(100, color_y, 200, 30)
            
            color_ui = Colors.GREEN if color == self.selected_color else Colors.LIGHT_GRAY
            pygame.draw.rect(self.screen, color_ui, button_rect)
            pygame.draw.rect(self.screen, Colors.BLACK, button_rect, 2)
            
            text = self.font_small.render(name, True, Colors.BLACK)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
    
    def _draw_game(self):
        """绘制游戏界面"""
        # 绘制棋盘
        self._draw_board()
        
        # 绘制UI面板
        self._draw_ui_panel()
        
        # 绘制最后一步动画
        if self.last_move_pos and time.time() - self.animation_time < self.animation_duration:
            self._draw_last_move_animation()
    
    def _draw_board(self):
        """绘制棋盘"""
        # 棋盘背景
        board_rect = pygame.Rect(self.margin, self.margin, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, Colors.LIGHT_BROWN, board_rect)
        
        # 绘制网格线
        for i in range(self.board_size):
            # 垂直线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.margin + self.board_height)
            pygame.draw.line(self.screen, Colors.BLACK, start_pos, end_pos, 1)
            
            # 水平线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.margin + self.board_width, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, Colors.BLACK, start_pos, end_pos, 1)
        
        # 绘制天元和星位
        center = self.board_size // 2
        star_positions = [(center, center)]
        
        for row, col in star_positions:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, Colors.BLACK, (x, y), 3)
        
        # 绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board.board[row, col] != self.board.EMPTY:
                    self._draw_stone(row, col, self.board.board[row, col])
    
    def _draw_stone(self, row: int, col: int, player: int):
        """绘制棋子"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        
        # 棋子颜色
        if player == self.board.BLACK:
            color = Colors.BLACK_STONE
            border_color = Colors.BLACK_STONE_BORDER
        else:
            color = Colors.WHITE_STONE
            border_color = Colors.WHITE_STONE_BORDER
        
        # 绘制棋子
        pygame.draw.circle(self.screen, color, (x, y), self.stone_radius)
        pygame.draw.circle(self.screen, border_color, (x, y), self.stone_radius, 2)
    
    def _draw_last_move_animation(self):
        """绘制最后一步动画"""
        if not self.last_move_pos:
            return
        
        row, col = self.last_move_pos
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        
        # 计算动画进度
        progress = (time.time() - self.animation_time) / self.animation_duration
        if progress > 1:
            progress = 1
        
        # 绘制高亮圆圈
        radius = int(self.stone_radius * (1 + progress * 0.5))
        alpha = int(255 * (1 - progress))
        
        # 创建半透明表面
        highlight_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(highlight_surface, (*Colors.YELLOW, alpha), (radius, radius), radius)
        
        self.screen.blit(highlight_surface, (x - radius, y - radius))
    
    def _draw_ui_panel(self):
        """绘制UI面板"""
        panel_x = self.board_width + 2 * self.margin
        panel_width = self.ui_width - 20
        
        # 面板背景
        panel_rect = pygame.Rect(panel_x, self.margin, panel_width, self.board_height)
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, Colors.BLACK, panel_rect, 2)
        
        # 游戏信息
        y_offset = 20
        
        # 当前玩家
        current_player_text = "黑子" if self.board.current_player == self.board.BLACK else "白子"
        player_text = self.font_medium.render(f"当前玩家: {current_player_text}", True, Colors.BLACK)
        self.screen.blit(player_text, (panel_x + 10, self.margin + y_offset))
        y_offset += 40
        
        # 显示人类玩家颜色
        human_color_text = "黑子" if self.human_player == self.board.BLACK else "白子"
        human_text = self.font_small.render(f"你执: {human_color_text}", True, Colors.BLACK)
        self.screen.blit(human_text, (panel_x + 10, self.margin + y_offset))
        y_offset += 30
        
        # 游戏模式
        mode_text = self.font_small.render(f"模式: {self.game_mode}", True, Colors.BLACK)
        self.screen.blit(mode_text, (panel_x + 10, self.margin + y_offset))
        y_offset += 30
        
        # AI信息
        if self.ai_player:
            ai_text = self.font_small.render(f"AI: AlphaZero", True, Colors.BLACK)
            self.screen.blit(ai_text, (panel_x + 10, self.margin + y_offset))
            y_offset += 30
            
            diff_text = self.font_small.render(f"难度: {self.difficulty}", True, Colors.BLACK)
            self.screen.blit(diff_text, (panel_x + 10, self.margin + y_offset))
            y_offset += 30
        
        # 步数
        move_count_text = self.font_small.render(f"步数: {self.board.get_move_count()}", True, Colors.BLACK)
        self.screen.blit(move_count_text, (panel_x + 10, self.margin + y_offset))
        y_offset += 40
        
        # 游戏状态
        if self.board.game_over:
            if self.board.winner:
                if self.game_mode == "human_vs_ai":
                    if self.board.winner == self.human_player:
                        winner_text = "你获胜了！"
                        status_color = Colors.GREEN
                    else:
                        winner_text = "AI获胜！"
                        status_color = Colors.RED
                else:
                    winner_text = "黑子获胜" if self.board.winner == self.board.BLACK else "白子获胜"
                    status_color = Colors.RED
                status_text = self.font_medium.render(winner_text, True, status_color)
            else:
                status_text = self.font_medium.render("平局", True, Colors.BLUE)
        else:
            status_text = self.font_medium.render("游戏中", True, Colors.GREEN)
        
        self.screen.blit(status_text, (panel_x + 10, self.margin + y_offset))
        y_offset += 40
        
        # 游戏结束提示
        if self.board.game_over:
            end_text = self.font_small.render("点击任意位置返回主菜单", True, Colors.DARK_GRAY)
            self.screen.blit(end_text, (panel_x + 10, self.margin + y_offset))
            y_offset += 30
        
        # 控制按钮
        button_height = 30
        button_spacing = 40
        
        buttons = [
            ("重新开始 (R)", Colors.BLUE),
            ("悔棋 (U)", Colors.GREEN),
            ("主菜单 (M)", Colors.RED),
            ("设置 (S)", Colors.GRAY)
        ]
        
        for text, color in buttons:
            button_rect = pygame.Rect(panel_x + 10, self.margin + y_offset, panel_width - 20, button_height)
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, Colors.BLACK, button_rect, 2)
            
            button_text = self.font_small.render(text, True, Colors.WHITE)
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, button_text_rect)
            
            y_offset += button_spacing
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        ai_move_timer = 0
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if not self.handle_key(event.key):
                        if event.key == pygame.K_ESCAPE:
                            running = False
            
            # AI对战模式：自动下棋
            if (self.game_started and self.game_mode == "ai_vs_ai" and 
                not self.board.game_over and current_time - ai_move_timer > 1000):
                self._make_ai_move()
                ai_move_timer = current_time
            
            # 绘制界面
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()


def test_ui():
    """测试UI"""
    print("Testing Gomoku UI...")
    
    ui = GomokuUI()
    
    # 设置回调
    def on_move_made(move, player):
        print(f"Move made: {move} by player {player}")
    
    def on_game_over(winner):
        if winner:
            print(f"Game over! Winner: {winner}")
        else:
            print("Game over! Draw!")
    
    ui.set_callbacks(on_move_made, on_game_over)
    
    # 运行UI
    ui.run()


if __name__ == "__main__":
    test_ui()
