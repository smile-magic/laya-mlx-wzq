import unittest

from game import BLACK, WHITE, SIZE, candidates, features, replay, select_move, threat_points, winning_line
from scenarios import positions


def empty():
    return [[0] * SIZE for _ in range(SIZE)]


class RulesTest(unittest.TestCase):
    def test_wins_in_all_four_directions_and_overlines(self):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for count in (5, 6):
                with self.subTest(direction=(dr, dc), count=count):
                    board = empty()
                    for i in range(count):
                        board[3 + i * dr][8 + i * dc] = BLACK
                    self.assertEqual(len(winning_line(board, 3, 8)), count)

    def test_four_is_not_a_win_and_edges_do_not_wrap(self):
        board = empty()
        for c in (0, 1, 2, 14):
            board[0][c] = BLACK
        self.assertFalse(winning_line(board, 0, 0))

    def test_replay_alternates_colors(self):
        board, winner, _ = replay([{"r": 7, "c": 7}, {"r": 7, "c": 8}])
        self.assertEqual((board[7][7], board[7][8]), (BLACK, WHITE))
        self.assertIsNone(winner)

    def test_rejects_invalid_inputs(self):
        for moves in (None, {}, [{"r": True, "c": 0}], [{"r": -1, "c": 0}],
                      [{"r": 0, "c": 15}], [{"r": 0, "c": 1.5}], [{}],
                      [{"r": 0, "c": 0}, {"r": 0, "c": 0}]):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                replay(moves)

    def test_rejects_moves_after_winning(self):
        moves = []
        for c in range(4):
            moves.extend([{"r": 7, "c": c}, {"r": 10, "c": c}])
        moves.append({"r": 7, "c": 4})
        self.assertEqual(replay(moves)[1], BLACK)
        with self.assertRaises(ValueError):
            replay(moves + [{"r": 2, "c": 2}])


class TacticsTest(unittest.TestCase):
    def test_winning_points_match_independent_board_scan(self):
        for _, symmetry, moves, _ in positions():
            if symmetry != 0:
                continue
            board, _, _ = replay(moves)
            for option in candidates(board):
                r, c = option["r"], option["c"]
                for color in (BLACK, WHITE):
                    win, expected = threat_points(board, r, c, color)
                    board[r][c] = color
                    self.assertEqual(win, bool(winning_line(board, r, c)))
                    if not win:
                        actual = set()
                        for nr in range(SIZE):
                            for nc in range(SIZE):
                                if board[nr][nc]:
                                    continue
                                board[nr][nc] = color
                                if (r, c) in winning_line(board, nr, nc):
                                    actual.add((nr, nc))
                                board[nr][nc] = 0
                        self.assertEqual(expected, actual)
                    board[r][c] = 0

    def test_opponent_immediate_win_overrides_own_future_fork(self):
        board = empty()
        board[7][3] = WHITE
        for c in range(4, 8):
            board[7][c] = BLACK
        for c in range(5, 8):
            board[10][c] = WHITE
        options = candidates(board)
        eligible = {(x["r"], x["c"]) for x in options if x["eligible"]}
        self.assertEqual(eligible, {(7, 8)})

    def test_forced_opponent_block_can_itself_create_a_fork(self):
        board = empty()
        for c in (4, 5, 6):
            board[7][c] = WHITE
        for r in (4, 5, 6):
            board[r][7] = BLACK
        options = candidates(board)
        # White I8 threatens H8, but Black's mandatory H8 creates a vertical
        # open four. This is a losing attack, despite forcing a response.
        trap = next(x for x in options if (x["r"], x["c"]) == (7, 8))
        self.assertEqual(trap["opponent_forks"], 1)
        self.assertFalse(trap["eligible"])

    def test_strong_shape_cannot_be_discarded_for_passive_move(self):
        board = empty()
        board[7][6] = board[7][7] = WHITE
        board[0][0] = BLACK
        options = candidates(board)
        best = max(x["score"] for x in options if x["tier"] == 2)
        self.assertTrue(all(x["score"] >= best * .65 for x in options if x["eligible"]))
        self.assertTrue(all(x["attack"].open_threes for x in options if x["eligible"]))

    def test_diagnostic_positions_reject_bad_model_preferences(self):
        for name, symmetry, moves, expected in positions():
            with self.subTest(case=name, symmetry=symmetry):
                board, winner, _ = replay(moves)
                self.assertIsNone(winner)
                options = candidates(board)
                self.assertTrue(expected.intersection((x["r"], x["c"]) for x in options))
                # Give every inferior candidate the strongest possible model preference.
                for bad in options:
                    if (bad["r"], bad["c"]) in expected:
                        continue
                    probabilities = {x["label"]: .001 for x in options}
                    probabilities[bad["label"]] = .99
                    _, executed, _ = select_move(options, probabilities)
                    self.assertIn((executed["r"], executed["c"]), expected)

    def test_immediate_win_and_broken_four(self):
        board = empty()
        for c in (4, 5, 7, 8):
            board[7][c] = WHITE
        self.assertTrue(features(board, 7, 6, WHITE).win)

    def test_open_four_has_two_winning_points(self):
        board = empty()
        for c in (5, 6, 7):
            board[7][c] = WHITE
        self.assertEqual(features(board, 7, 8, WHITE).winning_points, 2)

    def test_closed_four_has_one_winning_point(self):
        board = empty()
        board[7][4] = BLACK
        for c in (5, 6, 7):
            board[7][c] = WHITE
        self.assertEqual(features(board, 7, 8, WHITE).winning_points, 1)

    def test_open_three_is_detected(self):
        board = empty()
        for c in (6, 7):
            board[7][c] = WHITE
        self.assertGreater(features(board, 7, 8, WHITE).open_threes, 0)

    def test_analysis_does_not_mutate_board(self):
        board = empty()
        board[7][7] = BLACK
        original = [row[:] for row in board]
        candidates(board)
        self.assertEqual(board, original)

    def test_candidates_are_unique_empty_and_bounded(self):
        board = empty()
        board[7][7] = BLACK
        options = candidates(board)
        self.assertEqual(len(options), 6)
        self.assertEqual(len({x["label"] for x in options}), 6)
        self.assertTrue(all(board[x["r"]][x["c"]] == 0 for x in options))

    def test_rule_blocks_loss_even_when_model_prefers_other_move(self):
        board = empty()
        board[7][3] = WHITE
        for c in range(4, 8):
            board[7][c] = BLACK
        options = candidates(board)
        probabilities = {x["label"]: .01 for x in options}
        other = next(x for x in options if not x["defense"].win)
        probabilities[other["label"]] = .95
        proposed, executed, _ = select_move(options, probabilities)
        self.assertEqual(proposed["label"], other["label"])
        self.assertEqual((executed["r"], executed["c"]), (7, 8))

    def test_win_has_priority_over_block(self):
        board = empty()
        for c in range(4, 8):
            board[7][c] = WHITE
            board[10][c] = BLACK
        options = candidates(board)
        probabilities = {x["label"]: .9 if x["defense"].win else .01 for x in options}
        _, executed, _ = select_move(options, probabilities)
        self.assertTrue(executed["attack"].win)

    def test_no_guard_override_in_normal_position(self):
        board = empty()
        board[7][7] = BLACK
        options = candidates(board)
        probabilities = {x["label"]: .1 for x in options}
        probabilities[options[-1]["label"]] = .5
        proposed, executed, _ = select_move(options, probabilities)
        self.assertEqual(proposed, executed)
        self.assertEqual(executed, options[-1])


if __name__ == "__main__":
    unittest.main()
