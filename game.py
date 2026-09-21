"""Freestyle Gomoku rules and small, explicit tactical features. No ML dependency."""
from dataclasses import dataclass

SIZE = 15
BLACK, WHITE = 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def inside(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def coordinate(r, c):
    return f"{chr(65 + c)}{r + 1}"


def winning_line(board, r, c):
    color = board[r][c]
    if not color:
        return []
    for dr, dc in DIRECTIONS:
        line = [(r, c)]
        for sign in (-1, 1):
            nr, nc = r + sign * dr, c + sign * dc
            while inside(nr, nc) and board[nr][nc] == color:
                line.append((nr, nc))
                nr, nc = nr + sign * dr, nc + sign * dc
        if len(line) >= 5:
            return sorted(line)
    return []


def replay(moves):
    if not isinstance(moves, list) or len(moves) > SIZE * SIZE:
        raise ValueError("落子记录格式错误。")
    board = [[0] * SIZE for _ in range(SIZE)]
    winner, line = None, []
    for i, move in enumerate(moves):
        if winner:
            raise ValueError("棋局已经结束，不能继续落子。")
        if not isinstance(move, dict) or set(move) != {"r", "c"}:
            raise ValueError("落子格式错误。")
        r, c = move["r"], move["c"]
        if type(r) is not int or type(c) is not int or not inside(r, c):
            raise ValueError("落子坐标超出棋盘。")
        if board[r][c]:
            raise ValueError("这个交叉点已经有棋子。")
        board[r][c] = BLACK if i % 2 == 0 else WHITE
        line = winning_line(board, r, c)
        if line:
            winner = board[r][c]
    return board, winner, line


def _winning_cells(line):
    """Empty points completing five in a window containing the proposed center."""
    cells = set()
    for start in range(5):
        window = line[start:start + 5]
        if window.count(1) == 4 and window.count(0) == 1:
            cells.add(start + window.index(0))
    return cells


@dataclass(frozen=True)
class Features:
    win: bool = False
    winning_points: int = 0
    open_threes: int = 0
    potential: int = 0

    @property
    def value(self):
        if self.win:
            return 1_000_000_000
        if self.winning_points >= 2:
            return 10_000_000 + self.winning_points * 10_000
        return self.winning_points * 120_000 + self.open_threes * 12_000 + self.potential

    @property
    def summary(self):
        if self.win:
            return "直接连五"
        if self.winning_points >= 2:
            return "制造双杀"
        if self.winning_points:
            return "形成冲四"
        if self.open_threes >= 2:
            return "形成双活三"
        if self.open_threes:
            return "形成活三"
        return "延伸棋形"


def features(board, r, c, color):
    """Evaluate a hypothetical placement without mutating the board.

    The nine-cell directional window captures wins, broken fours and open
    threes. These are local features, not a game-tree proof of a forced win.
    """
    winning, threats, threes, potential = False, 0, 0, 0
    for dr, dc in DIRECTIONS:
        line = []
        for offset in range(-4, 5):
            nr, nc = r + offset * dr, c + offset * dc
            cell = board[nr][nc] if inside(nr, nc) else 3
            line.append(1 if cell == color else 0 if cell == 0 else 2)
        line[4] = 1
        winning |= any(line[start:start + 5] == [1] * 5 for start in range(5))
        completions = _winning_cells(line)
        threats += len(completions)
        if not completions:
            for i, cell in enumerate(line):
                if cell == 0:
                    trial = line.copy()
                    trial[i] = 1
                    if len(_winning_cells(trial)) >= 2:
                        threes += 1
                        break
        for start in range(5):
            window = line[start:start + 5]
            if 2 not in window:
                potential += (0, 1, 8, 50, 200, 1000)[window.count(1)]
    return Features(winning, threats, threes, potential)


def threat_points(board, r, c, color):
    """Immediate win and distinct future winning squares after a placement.

    Only lines through the hypothetical stone can create new winning squares.
    Existing threats elsewhere are combined by the caller. No board mutation.
    """
    wins, points = False, set()
    for dr, dc in DIRECTIONS:
        line = []
        for offset in range(-4, 5):
            nr, nc = r + offset * dr, c + offset * dc
            cell = board[nr][nc] if inside(nr, nc) else 3
            line.append(1 if cell == color else 0 if cell == 0 else 2)
        line[4] = 1
        wins |= any(line[start:start + 5] == [1] * 5 for start in range(5))
        points.update((r + (i - 4) * dr, c + (i - 4) * dc) for i in _winning_cells(line))
    return wins, points


def candidates(board):
    occupied = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c]]
    nearby = set()
    for r, c in occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if inside(nr, nc) and not board[nr][nc]:
                    nearby.add((nr, nc))
    if not occupied:
        nearby.add((7, 7))
    result = []
    for r, c in sorted(nearby):
        attack, defense = features(board, r, c, WHITE), features(board, r, c, BLACK)
        score = max(attack.value, defense.value * 0.92) + min(attack.value, defense.value) * 0.05
        score -= (abs(r - 7) + abs(c - 7)) * 0.1
        result.append({"r": r, "c": c, "label": coordinate(r, c),
                       "attack": attack, "defense": defense, "score": score})
    if not result:
        return []
    white_wins = {(x["r"], x["c"]) for x in result if x["attack"].win}
    black_wins = {(x["r"], x["c"]) for x in result if x["defense"].win}
    black_forks = {(x["r"], x["c"]) for x in result if x["defense"].winning_points >= 2}

    # Inspect the entire neighborhood BEFORE the six-option model shortlist.
    # Adding a white stone cannot create a new black win/fork, so only the
    # original black threat locations need rechecking after each white move.
    for option in result:
        r, c = option["r"], option["c"]
        _, created = threat_points(board, r, c, WHITE)
        threats = (white_wins | created) - {(r, c)}
        remaining_wins = black_wins - {(r, c)}
        forks = 0
        board[r][c] = WHITE
        try:
            if not option["attack"].win and not remaining_wins and len(threats) < 2:
                # A single white winning square forces Black to occupy it.
                # Other black fork moves would lose to White's immediate win.
                replies = threats if threats else black_forks
                for br, bc in replies:
                    if board[br][bc]:
                        continue
                    _, points = threat_points(board, br, bc, BLACK)
                    forks += len(points) >= 2
        finally:
            board[r][c] = 0

        if option["attack"].win:
            tier, tactic = 4, "直接连五"
        elif remaining_wins:
            tier, tactic = 0, "对手下一手可连五"
        elif len(threats) >= 2:
            tier, tactic = 3, "形成双重连五点"
        elif forks:
            tier, tactic = 1, "对手可制造双重连五点"
        else:
            tier, tactic = 2, "未检出短程败招"
        option.update(tier=tier, tactic=tactic, opponent_wins=len(remaining_wins),
                      opponent_forks=forks, future_wins=len(threats))

    best_tier = max(x["tier"] for x in result)
    best_score = max(x["score"] for x in result if x["tier"] == best_tier)
    # Within ordinary positions, don't let a model preference discard a
    # clearly stronger shape. This is explicitly a heuristic, not a proof.
    threshold = best_score * .65 if best_tier == 2 and best_score > 0 else -float("inf")
    if best_tier == 4:
        reason = "完成连五"
    elif best_tier == 3:
        reason = "形成对手无法同时封堵的双重连五点"
    elif best_tier < 2:
        reason = "候选中未找到可解除全部短程威胁的走法"
    elif black_wins:
        reason = "拦截对方下一手连五"
    elif black_forks:
        reason = "提前防住对方活四或双重连五威胁"
    else:
        reason = "在短程检查通过、棋形评分相近的候选中选择"
    for option in result:
        option["eligible"] = option["tier"] == best_tier and option["score"] >= threshold
        option["guard_reason"] = reason
    result.sort(key=lambda x: (x["eligible"], x["tier"], x["score"]), reverse=True)
    return result[:6]


def select_move(options, probabilities):
    proposed = max(options, key=lambda x: probabilities[x["label"]])
    allowed = [x for x in options if x["eligible"]]
    executed = max(allowed, key=lambda x: probabilities[x["label"]])
    return proposed, executed, executed["guard_reason"]
