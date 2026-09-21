import unittest

from game import BLACK, WHITE, SIZE, candidates, features, replay, select_move, winning_line


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
