"""Tests de l'exercice 3 : save_state / restore_state, GreedyPlayer et programme principal.

Lancement depuis la racine du dépôt : python -m unittest discover tests
"""

import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ex3_qlearning import (
    moving_average, paired_difference, play, run, run_many, t95, total_stats, verdict,
)
from greedy_player import GreedyPlayer
from mdp import actions, state_of
from snake import DOWN, EMPTY, FRUIT, LEFT, RIGHT, SNAKE, UP, ShortSnake


def count_cells(game, value):
    return sum(row.count(value) for row in game.grid)


class TestSaveRestore(unittest.TestCase):
    def test_save_state_is_mdp_state(self):
        game = ShortSnake(10, seed=0)
        state = game.save_state()
        self.assertIsInstance(state, tuple)
        self.assertEqual(state, ((5, 5), RIGHT, game.fruit_pos))
        self.assertEqual(state, state_of(game))

    def test_state_is_hashable(self):
        game = ShortSnake(10, seed=0)
        self.assertEqual({game.save_state(): 1}[game.save_state()], 1)

    def test_restore_previous_state(self):
        game = ShortSnake(10, seed=0)
        saved = game.save_state()
        for direction in (UP, LEFT, LEFT, DOWN, None):
            game.move(direction)
        self.assertNotEqual(game.save_state(), saved)
        game.restore_state(saved)
        self.assertEqual(game.save_state(), saved)
        self.assertEqual(count_cells(game, SNAKE), 1)
        self.assertEqual(count_cells(game, FRUIT), 1)
        self.assertEqual(game.grid[5][5], SNAKE)

    def test_restore_then_replay_is_identical(self):
        game = ShortSnake(6, seed=4)
        saved = ((2, 5), RIGHT, (2, 0))
        game.restore_state(saved)
        game.rng.seed(7)
        first = (game.move(), game.save_state())   # bord franchi (-10) et fruit mangé (+10)
        game.restore_state(saved)
        game.rng.seed(7)
        self.assertEqual((game.move(), game.save_state()), first)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1][0], (2, 0))

    def test_restore_rejects_invalid_states(self):
        game = ShortSnake(4, seed=0)
        for state in (((4, 0), UP, (0, 0)), ((0, 0), "DIAG", (1, 1)), ((1, 1), UP, (1, 1))):
            with self.assertRaises(ValueError):
                game.restore_state(state)

    def test_restore_normalizes_positions(self):
        game = ShortSnake(4, seed=0)
        game.restore_state(([1, 2], UP, np.array([3, 0])))
        state = game.save_state()
        self.assertEqual(state, ((1, 2), UP, (3, 0)))
        self.assertIsInstance(state[0], tuple)
        hash(state)

    def test_invalid_restore_leaves_game_untouched(self):
        game = ShortSnake(4, seed=0)
        before = (game.save_state(), [row[:] for row in game.grid])
        for state in (((0.5, 0), UP, (1, 1)), ((1, 1), UP, [1, 1]), (("a", 0), UP, (1, 1)),
                      ((1, 2, 3), UP, (1, 1)), None, ((1, 1), ["UP"], (2, 2)),
                      ((1, 1), UP)):
            with self.assertRaises(ValueError):
                game.restore_state(state)
            self.assertEqual((game.save_state(), game.grid), before)

    def test_restore_keeps_score(self):
        game = ShortSnake(4, seed=0)
        game.score = 30
        game.restore_state(((0, 0), UP, (3, 3)))
        self.assertEqual(game.score, 30)
        self.assertEqual(game.grid[0][1], EMPTY)


class TestGreedyPlayer(unittest.TestCase):
    def setUp(self):
        self.game = ShortSnake(10, seed=0)
        self.game.restore_state(((5, 5), RIGHT, (0, 0)))

    def test_init(self):
        player = GreedyPlayer(self.game, 0.1)
        self.assertIs(player.game, self.game)
        self.assertEqual(player.action_values, {})
        self.assertEqual((player.alpha, player.gamma), (0.5, 0.5))

    def test_invalid_parameters(self):
        for kwargs in ({"eps": -0.1}, {"eps": 1.5}, {"eps": 0.1, "alpha": 0},
                       {"eps": 0.1, "gamma": 1.1}):
            with self.assertRaises(ValueError):
                GreedyPlayer(self.game, **kwargs)

    def test_actions_are_legal(self):
        player = GreedyPlayer(self.game, 0.5, seed=1)
        for _ in range(500):
            state = self.game.save_state()
            action = player.get_action()
            self.assertIn(action, actions(state))
            player.reward(action, self.game.move(action))

    def test_greedy_uses_current_state(self):
        player = GreedyPlayer(self.game, 0, seed=1)
        state = self.game.save_state()
        player.action_values[(state, UP)] = 3.0
        player.action_values[(state, DOWN)] = -1.0
        other = ((5, 5), RIGHT, (9, 9))
        player.action_values[(other, DOWN)] = 100.0  # autre état : ne doit pas compter
        self.assertEqual(player.get_action(), UP)

    def test_greedy_ties_are_random(self):
        player = GreedyPlayer(self.game, 0, seed=1)
        chosen = {player._greedy_action(self.game.save_state()) for _ in range(100)}
        self.assertEqual(chosen, {UP, DOWN, RIGHT})

    def test_full_exploration(self):
        player = GreedyPlayer(self.game, 1, seed=1)
        state = self.game.save_state()
        player.action_values[(state, UP)] = 50.0
        chosen = {player.get_action() for _ in range(100)}
        self.assertEqual(chosen, {UP, DOWN, RIGHT})

    def test_q_learning_update(self):
        alpha, gamma = 0.3, 0.8
        player = GreedyPlayer(self.game, 0, alpha=alpha, gamma=gamma, seed=1)
        state = self.game.save_state()
        next_state = ((5, 6), RIGHT, (0, 0))
        player.action_values[(state, RIGHT)] = 2.0
        player.action_values[(next_state, UP)] = 4.0
        player.action_values[(next_state, DOWN)] = -6.0
        action = player.get_action()
        self.assertEqual(action, RIGHT)
        value = self.game.move(action)
        player.reward(action, value)
        expected = (1 - alpha) * 2.0 + alpha * (0 + gamma * 4.0)
        self.assertAlmostEqual(player.action_values[(state, RIGHT)], expected)
        self.assertEqual(player.eval_count[(state, RIGHT)], 1)

    def test_update_with_wall_penalty_and_gamma_zero(self):
        self.game.restore_state(((5, 9), RIGHT, (0, 0)))
        player = GreedyPlayer(self.game, 0, alpha=0.5, gamma=0, seed=1)
        state = self.game.save_state()
        player.action_values[(state, RIGHT)] = 1.0
        player.action_values[(((5, 0), RIGHT, (0, 0)), UP)] = 99.0  # ignoré car gamma = 0
        action = player.get_action()
        player.reward(action, self.game.move(action))
        self.assertAlmostEqual(player.action_values[(state, RIGHT)], 0.5 * 1.0 + 0.5 * -10)

    def test_reward_needs_get_action(self):
        player = GreedyPlayer(self.game, 0.1)
        with self.assertRaises(RuntimeError):
            player.reward(UP, 0)

    def test_one_update_per_action(self):
        player = GreedyPlayer(self.game, 0.1, seed=1)
        action = player.get_action()
        player.reward(action, self.game.move(action))
        with self.assertRaises(RuntimeError):
            player.reward(action, 0)

    def test_reward_rejects_impossible_action(self):
        player = GreedyPlayer(self.game, 0.1)
        player.get_action()
        with self.assertRaises(ValueError):
            player.reward(LEFT, 0)   # demi-tour


class TestMainProgram(unittest.TestCase):
    def test_play_returns_rewards(self):
        game = ShortSnake(10, seed=0)
        player = GreedyPlayer(game, 0.1, seed=1)
        rewards = play(player, game, 300)
        self.assertEqual(rewards.shape, (300,))
        self.assertEqual(int(rewards.sum()), game.score)
        self.assertTrue(set(np.unique(rewards)) <= {-10, 0, 10})

    def test_run_is_reproducible(self):
        np.testing.assert_array_equal(run(0.5, 500, seed=3), run(0.5, 500, seed=3))

    def test_learning_beats_random(self):
        learned = run(0.5, 100_000, seed=0)[-10_000:].mean()
        random_play = run(0.5, 100_000, seed=0, eps=1.0)[-10_000:].mean()
        self.assertGreater(learned, random_play + 0.3)

    def test_run_many_keeps_seed_order(self):
        with ProcessPoolExecutor(2) as pool:
            rewards = run_many(pool, 0.5, 200, 5, first_seed=10)
        self.assertEqual(rewards.shape, (5, 200))
        for i in range(5):
            np.testing.assert_array_equal(rewards[i], run(0.5, 200, seed=10 + i))

    def test_run_many_rejects_empty(self):
        with self.assertRaises(ValueError):
            run_many(None, 0.5, 100, 0)


class TestStatistics(unittest.TestCase):
    def test_t95(self):
        exact = {1: 12.706, 3: 3.182, 6: 2.447, 7: 2.365, 9: 2.262, 31: 2.040, 999: 1.962}
        for df, value in exact.items():
            self.assertAlmostEqual(t95(df), value, delta=0.005 * value)

    def test_verdict(self):
        self.assertEqual(verdict(0, 0), "aucun écart")
        self.assertEqual(verdict(-0.13, 0.26), "non significatif")
        self.assertEqual(verdict(5.0, 1.0), "significatif")
        self.assertEqual(verdict(5.0, float("nan")), "indéterminé")

    def test_total_stats(self):
        rewards = np.array([[10, 0, -10], [10, 10, 0], [0, 0, 0], [-10, -10, 0]], dtype=np.int8)
        totals = np.array([0, 20, 0, -20])
        mean, ci = total_stats(rewards)
        self.assertEqual(mean, 0)
        self.assertAlmostEqual(ci, t95(3) * totals.std(ddof=1) / 2)
        self.assertTrue(np.isnan(total_stats(rewards[:1])[1]))

    def test_paired_difference(self):
        a = np.array([[10, 0], [0, -10], [10, 10]], dtype=np.int8)
        self.assertEqual(paired_difference(a, a), (0, 0))
        b = a.copy()
        b[1, 1] = 0
        diff, _ = paired_difference(b, a)
        self.assertAlmostEqual(diff, 10 / 3)

    def test_moving_average_is_trailing(self):
        values = np.zeros(60)
        values[39] = 10                     # impulsion à l'étape 40
        smoothed = moving_average(values, 10)
        steps = np.arange(1, 61)[9:]        # alignement utilisé dans les graphiques
        self.assertEqual(list(steps[smoothed > 0]), list(range(40, 50)))


if __name__ == "__main__":
    unittest.main()
