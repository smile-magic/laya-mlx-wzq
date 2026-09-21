"""Small legal tactical positions from the decision diagnosis, not game logs."""
CASES = {
    "create_open_four": (
        [(0, 0), (7, 5), (0, 14), (7, 6), (14, 0), (7, 7), (14, 14)],
        {(7, 4), (7, 8)},
    ),
    "defend_open_three": (
        [(7, 5), (1, 1), (7, 6), (1, 3), (7, 7), (2, 9), (10, 10)],
        {(7, 4), (7, 8)},
    ),
    "block_win": (
        [(7, 4), (7, 3), (7, 5), (0, 0), (7, 6), (0, 2), (7, 7)],
        {(7, 8)},
    ),
    "take_win": (
        [(10, 3), (7, 4), (10, 4), (7, 5), (10, 5), (7, 6),
         (10, 6), (7, 7), (0, 0)],
        {(7, 3), (7, 8)},
    ),
}


def transform(point, symmetry):
    r, c = point
    if symmetry >= 4:
        c = 14 - c
    for _ in range(symmetry % 4):
        r, c = c, 14 - r
    return r, c


def positions():
    for name, (points, expected) in CASES.items():
        for symmetry in range(8):
            moves = [dict(zip(("r", "c"), transform(p, symmetry))) for p in points]
            yield name, symmetry, moves, {transform(p, symmetry) for p in expected}
