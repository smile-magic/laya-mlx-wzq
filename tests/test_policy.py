import os
import unittest
from pathlib import Path

from game import candidates, coordinate, replay
from scenarios import positions
from server import Policy, decision_prompt


class PolicyTest(unittest.TestCase):
    def test_guard_keeps_raw_probabilities_and_reports_the_actual_intervention(self):
        _, _, moves, expected = next(positions())
        board, _, _ = replay(moves)
        options = candidates(board)
        bad = next(x for x in options if not x["eligible"])
        values = {x["label"]: .01 for x in options}
        values[bad["label"]] = .95

        class FixedAgent:
            def predict(self, state, questions):
                return {"answers": {"move": {"probabilities": dict(values)}}}

        policy = Policy.__new__(Policy)
        policy.agent = FixedAgent()
        result = policy.decide(moves)
        analysis = result["analysis"]
        self.assertEqual(analysis["proposed"], bad["label"])
        self.assertIn((result["move"]["r"], result["move"]["c"]), expected)
        self.assertTrue(analysis["intervened"])
        self.assertEqual({x["label"]: x["probability"] for x in analysis["candidates"]}, values)
        chosen = next(x for x in analysis["candidates"] if x["label"] == analysis["executed"])
        self.assertTrue(chosen["eligible"])

    def test_prompt_contains_exact_board_and_each_candidate(self):
        _, _, moves, _ = next(positions())
        board, _, _ = replay(moves)
        options = candidates(board)
        state, questions = decision_prompt(board, moves, options)
        rows = state.splitlines()[-15:]
        for r, row in enumerate(rows):
            self.assertEqual(row, f"{r+1:02d} " + " ".join(".BW"[cell] for cell in board[r]))
        self.assertEqual(set(questions["move"]["criteria"]), {x["label"] for x in options})


@unittest.skipUnless(os.environ.get("LAYA_TEST_MODEL"), "Set LAYA_TEST_MODEL for real GPU tests")
class ModelRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = Policy(Path(os.environ["LAYA_TEST_MODEL"]).expanduser())

    def test_diagnostic_32_positions_with_real_model(self):
        for name, symmetry, moves, expected in positions():
            with self.subTest(case=name, symmetry=symmetry):
                result = self.policy.decide(moves)
                self.assertIn((result["move"]["r"], result["move"]["c"]), expected)
                self.assertEqual(result["analysis"]["executed"],
                                 coordinate(result["move"]["r"], result["move"]["c"]))

    def test_real_tokenizer_preserves_complete_board_and_option_descriptions(self):
        for _, _, moves, _ in positions():
            board, _, _ = replay(moves)
            options = candidates(board)
            state, questions = decision_prompt(board, moves, options)
            items, _ = self.policy.agent.prepare(state, questions)
            item = items[0]
            self.assertLess(len(item["ids"]), self.policy.agent.cfg["max_len"])
            self.assertEqual(len(item["markers"]), len(options))
            decoded = self.policy.agent.tok.backend.decode(item["ids"], skip_special_tokens=False)
            self.assertIn(state.splitlines()[-1], decoded)
            for description in questions["move"]["criteria"].values():
                self.assertIn(description, decoded)
