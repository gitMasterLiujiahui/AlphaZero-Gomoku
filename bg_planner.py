"""
BG-Planner Model for Gomoku

Implements three core components from the paper "A Novel Decision-Making Model for Playing Board Game Combining Planning and Opponent Behaviors" (2025):
1) GraphNet (core planner) with alternating action and proposition layers
2) Opponent Modeling via DQN to predict opponent behaviors
3) Knowledge-based Search using Gomoku domain heuristics (live four, rush four, double live three)

This is a practical, CPU-friendly implementation designed to integrate with the existing project.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from gomoku_board import GomokuBoard


class GraphNet(nn.Module):
    """
    Graph planning network with alternating action and proposition layers.
    For efficiency, we approximate board as a grid-graph with 4- or 8-neighborhood.
    """

    def __init__(self, board_size: int = 15, in_channels: int = 3, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim

        # Initial embedding from one-hot planes: black, white, empty
        self.embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        # Alternating action/proposition layers approximated by GCN-like convs
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))  # proposition-like
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))             # action-like
        self.layers = nn.ModuleList(layers)

        # Output policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.embed(x))
        for i in range(0, len(self.layers), 2):
            h = F.relu(self.layers[i](h))
            h = F.relu(self.layers[i + 1](h))
        policy_logits = self.policy_head(h)
        return policy_logits


class OpponentDQN(nn.Module):
    """
    Simple DQN to predict opponent move quality on flattened board.
    """

    def __init__(self, board_size: int = 15, in_channels: int = 3, hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        input_dim = in_channels * board_size * board_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, board_size * board_size)
        )

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        # planes: (B, C, H, W)
        x = planes.view(planes.size(0), -1)
        q = self.net(x)
        return q


class KnowledgeSearch:
    """
    Knowledge-based search using heuristic patterns for Gomoku.
    Provides a fast scoring for candidate moves and can short-circuit for tactical wins/blocks.
    """

    def __init__(self, board_size: int = 15):
        self.n = board_size

    def score_move(self, board: GomokuBoard, move: Tuple[int, int], player: int) -> float:
        # Simulate placing and evaluate patterns
        if not board.is_valid_move(move[0], move[1]):
            return -1e9
        temp = board.copy_board()
        temp.make_move(move[0], move[1])
        # Immediate win
        if temp.game_over and temp.winner == player:
            return 1e6
        # Block opponent immediate win
        opp = GomokuBoard.WHITE if player == GomokuBoard.BLACK else GomokuBoard.BLACK
        if self._opponent_can_win_next(temp, opp):
            return -1e5
        # Heuristic features: live four, rush four, double live three
        line_score = self._pattern_score(temp, player)
        center_bias = self._center_bias(move)
        return line_score + center_bias

    def top_k_moves(self, board: GomokuBoard, player: int, k: int = 10) -> List[Tuple[int, int]]:
        moves = board.get_valid_moves()
        if not moves:
            return []
        scored = [(self.score_move(board, m, player), m) for m in moves]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored[:k]]

    def _opponent_can_win_next(self, board: GomokuBoard, opponent: int) -> bool:
        for (r, c) in board.get_valid_moves():
            temp = board.copy_board()
            if temp.current_player != opponent:
                # Force opponent turn for prediction
                temp.current_player = opponent
            temp.make_move(r, c)
            if temp.game_over and temp.winner == opponent:
                return True
        return False

    def _center_bias(self, move: Tuple[int, int]) -> float:
        center = self.n // 2
        dr = abs(move[0] - center)
        dc = abs(move[1] - center)
        return max(0.0, (6 - (dr + dc)) * 0.5)

    def _pattern_score(self, board: GomokuBoard, player: int) -> float:
        # Lightweight pattern scoring over all lines through last move or entire board
        # Scores for common patterns
        scores = {
            'FIVE': 100000.0,
            'LIVE_FOUR': 10000.0,
            'RUSH_FOUR': 5000.0,
            'DOUBLE_LIVE_THREE': 3000.0,
            'LIVE_THREE': 1000.0,
            'RUSH_THREE': 300.0,
            'LIVE_TWO': 50.0,
        }
        arr = board.board
        total = 0.0
        dirs = [(1,0),(0,1),(1,1),(1,-1)]
        for r in range(self.n):
            for c in range(self.n):
                if arr[r,c] != player:
                    continue
                for dr, dc in dirs:
                    segment = self._extract_line(arr, r, c, dr, dc, length=9)
                    total += self._eval_segment(segment, scores, player)
        return total

    def _extract_line(self, arr: np.ndarray, r: int, c: int, dr: int, dc: int, length: int) -> List[int]:
        half = length // 2
        vals = []
        for k in range(-half, half+1):
            rr = r + k*dr
            cc = c + k*dc
            if 0 <= rr < self.n and 0 <= cc < self.n:
                vals.append(int(arr[rr, cc]))
            else:
                vals.append(-1)
        return vals

    def _eval_segment(self, seg: List[int], scores: Dict[str, float], player: int) -> float:
        s = ''.join({-1:'#',0:'.',1:'X',2:'O'}[v] for v in seg)
        me = 'X' if player == 1 else 'O'
        empty = '.'
        # Simple pattern checks
        val = 0.0
        if me*5 in s:
            val += scores['FIVE']
        # live four: .XXXX. (or boundary-adjacent approximations)
        if f"{empty}{me*4}{empty}" in s:
            val += scores['LIVE_FOUR']
        # rush four: XXXX. or .XXXX at boundary-like
        if f"{me*4}{empty}" in s or f"{empty}{me*4}" in s:
            val += scores['RUSH_FOUR']
        # live three: .XXX. with at least one open end beyond
        if f"{empty}{me*3}{empty}" in s:
            val += scores['LIVE_THREE']
        # rush three: patterns like XX.X or X.XX with an open end
        rush_three_patterns = [f"{me*2}{empty}{me}", f"{me}{empty}{me*2}"]
        if any(p in s for p in rush_three_patterns):
            val += scores['RUSH_THREE']
        # live two: .XX.
        if f"{empty}{me*2}{empty}" in s:
            val += scores['LIVE_TWO']
        # double live three (very rough): two live three occurrences
        if s.count(f"{empty}{me*3}{empty}") >= 2:
            val += scores['DOUBLE_LIVE_THREE']
        return val


class BGPlannerAI:
    """
    BG-Planner AI combining GraphNet, Opponent DQN and Knowledge-based Search.
    """

    def __init__(self, player: int, difficulty: str = "medium", device: str = "cpu"):
        self.player = player
        self.difficulty = difficulty
        self.device = torch.device(device)

        self.board_size = 15
        self.graph_net = GraphNet(self.board_size).to(self.device)
        self.opp_dqn = OpponentDQN(self.board_size).to(self.device)
        self.k_search = KnowledgeSearch(self.board_size)

        # Exploration/exploitation settings by difficulty
        self.params = {
            'easy':  {'k': 8,  'mix_alpha': 0.5, 'explore': 0.2},
            'medium':{'k': 12, 'mix_alpha': 0.65,'explore': 0.1},
            'hard':  {'k': 16, 'mix_alpha': 0.75,'explore': 0.05}
        }[difficulty if difficulty in ['easy','medium','hard'] else 'medium']

        # Random init; external training can load weights later if provided
        self.graph_net.eval()
        self.opp_dqn.eval()

    def _board_to_planes(self, board_state: np.ndarray) -> np.ndarray:
        planes = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        planes[0] = (board_state == 1).astype(np.float32)
        planes[1] = (board_state == 2).astype(np.float32)
        planes[2] = (board_state == 0).astype(np.float32)
        return planes

    def get_move(self, board: GomokuBoard) -> Optional[Tuple[int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None

        # Knowledge-based top-K candidates
        top_moves = self.k_search.top_k_moves(board, self.player, k=self.params['k'])
        if not top_moves:
            return random.choice(valid_moves)

        # GraphNet policy over all cells
        planes = self._board_to_planes(board.get_board_state())
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits = self.graph_net(x).squeeze(0)
            policy = torch.softmax(policy_logits, dim=0).cpu().numpy()

            # Opponent DQN estimates opponent interest; we use it to penalize risky cells
            q_opp = self.opp_dqn(x).squeeze(0).cpu().numpy()

        # Compose scores for candidates
        mix_alpha = self.params['mix_alpha']
        best_move = None
        best_score = -1e18
        for r, c in top_moves:
            idx = r * self.board_size + c
            net_score = float(policy[idx])
            opp_penalty = float(q_opp[idx])
            # lower opp_penalty implies safer; invert by negative scaling
            composed = mix_alpha * net_score - (1 - mix_alpha) * (opp_penalty)
            if composed > best_score:
                best_score = composed
                best_move = (r, c)

        # Exploration
        if random.random() < self.params['explore']:
            return random.choice(top_moves)
        return best_move or random.choice(valid_moves)


