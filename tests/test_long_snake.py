"""Tests de l'exercice 4 : serpent de longueur L ≥ 2.

Lancement depuis la racine du dépôt : python -m unittest discover tests
"""

import random
import unittest

from ex3_qlearning import play, run
from greedy_player import GreedyPlayer
from mdp import (
    MOVE_ARGUMENTS, actions, all_long_states, body_shapes, n_long_states, reachable_states,
    transitions,
)
from snake import DOWN, EMPTY, FRUIT, LEFT, RIGHT, SNAKE, UP, ShortSnake


def count_cells(game, value):
    return sum(row.count(value) for row in game.grid)


class TestLongSnakeGame(unittest.TestCase):
    def test_initial_body(self):
        game = ShortSnake(10, seed=0, length=3)
        self.assertEqual(game.snake_pos, (5, 5))
        self.assertEqual(list(game.body), [(5, 4), (5, 3)])
        self.assertEqual(count_cells(game, SNAKE), 3)
        self.assertNotIn(game.fruit_pos, [game.snake_pos, *game.body])

    def test_invalid_length(self):
        for length in (0, 11):
            with self.assertRaises(ValueError):
                ShortSnake(10, length=length)
        # plus long que le petit côté : il pourrait se recouvrir en ligne droite
        for size, length in (((1, 6), 2), ((2, 5), 3), ((3, 10), 4)):
            with self.assertRaises(ValueError):
                ShortSnake(size, length=length)
        ShortSnake((2, 5), length=2)
        ShortSnake((3, 10), length=3)

    def test_length_one_is_unchanged(self):
        self.assertEqual(int(run(0.5, 1000, seed=0).sum()), -870)   # partie de la question 3.10

    def test_body_follows_head(self):
        game = ShortSnake(10, seed=0, length=3)
        game.restore_state(((5, 5), RIGHT, (0, 0), ((5, 4), (5, 3))))
        self.assertEqual(game.move(UP), 0)
        self.assertEqual(game.snake_pos, (4, 5))
        self.assertEqual(list(game.body), [(5, 5), (5, 4)])
        self.assertEqual(game.grid[5][3], EMPTY)
        self.assertEqual(count_cells(game, SNAKE), 3)

    def test_no_u_turn(self):
        game = ShortSnake(10, seed=0, length=2)
        game.move(LEFT)   # demi-tour : ignoré, sinon la tête entrerait dans le cou
        self.assertEqual(game.snake_pos, (5, 6))
        self.assertEqual(game.self_hits, 0)

    def test_body_wraps_around(self):
        game = ShortSnake(10, seed=0, length=3)
        game.restore_state(((5, 9), RIGHT, (0, 0), ((5, 8), (5, 7))))
        self.assertEqual(game.move(), -10)
        self.assertEqual(game.save_state(), ((5, 0), RIGHT, (0, 0), ((5, 9), (5, 8))))

    def test_no_self_hit_below_length_5(self):
        game = ShortSnake(10, seed=0, length=4)
        game.restore_state(((5, 5), RIGHT, (0, 0), ((5, 4), (5, 3), (5, 2))))
        rewards = [game.move(d) for d in (DOWN, LEFT, UP)]   # la tête revient en (5, 4)
        self.assertEqual(rewards, [0, 0, 0])
        self.assertEqual(game.self_hits, 0)

    def test_self_hit_from_length_5(self):
        game = ShortSnake(10, seed=0, length=5)
        game.restore_state(((5, 5), RIGHT, (0, 0), ((5, 4), (5, 3), (5, 2), (5, 1))))
        rewards = [game.move(d) for d in (DOWN, LEFT, UP)]
        self.assertEqual(rewards, [0, 0, -10])
        self.assertEqual(game.self_hits, 1)
        self.assertEqual(game.snake_pos, (5, 4))
        self.assertIn((5, 4), game.body)
        # la case occupée deux fois reste occupée quand un des deux morceaux s'en va
        self.assertEqual(count_cells(game, SNAKE), len({game.snake_pos, *game.body}))
        for _ in range(4):
            game.move(UP)
        self.assertEqual(count_cells(game, SNAKE), len({game.snake_pos, *game.body}))
        self.assertEqual(count_cells(game, FRUIT), 1)

    def test_fruit_never_on_body(self):
        game = ShortSnake(4, seed=3, length=3)
        rng = random.Random(0)
        for _ in range(3000):
            game.move(rng.choice(MOVE_ARGUMENTS))
            self.assertNotIn(game.fruit_pos, [game.snake_pos, *game.body])
            self.assertEqual(count_cells(game, FRUIT), 1)

    def test_save_restore_roundtrip(self):
        game = ShortSnake(6, seed=1, length=4)
        rng = random.Random(2)
        for _ in range(200):
            game.move(rng.choice(MOVE_ARGUMENTS))
            state = game.save_state()
            self.assertEqual(len(state), 4)
            other = ShortSnake(6, seed=9, length=4)
            other.restore_state(state)
            self.assertEqual(other.save_state(), state)
            self.assertEqual(other.grid, game.grid)

    def test_restore_rejects_bad_body(self):
        game = ShortSnake(10, seed=0, length=3)
        before = (game.save_state(), [row[:] for row in game.grid])
        for state in (((5, 5), RIGHT, (0, 0), ((5, 4),)),              # mauvaise longueur
                      ((5, 5), RIGHT, (0, 0), ((4, 5), (3, 5))),        # cou pas derrière la tête
                      ((5, 5), RIGHT, (0, 0), ((5, 4), (7, 4))),        # cases non voisines
                      ((5, 5), RIGHT, (5, 3), ((5, 4), (5, 3))),        # fruit sur le corps
                      ((5, 5), RIGHT, (0, 0), ((5, 4), (5, 5))),        # demi-tour dans le corps
                      ((5, 5), RIGHT, (0, 0), ()),                       # corps vide
                      ((5, 5), RIGHT, (0, 0))):                          # corps absent
            with self.assertRaises(ValueError):
                game.restore_state(state)
            self.assertEqual((game.save_state(), game.grid), before)

    def test_str_shows_body(self):
        game = ShortSnake((3, 5), seed=0, length=3)
        self.assertIn("oo>", str(game))


class TestLongSnakeModel(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual([len(body_shapes(n)) for n in (1, 2, 3, 4, 5)], [4, 4, 12, 36, 108])

    def test_state_counts(self):
        self.assertEqual(n_long_states(10, 10, 1), 39600)
        self.assertEqual(n_long_states(10, 10, 2), 4 * 100 * 98)
        self.assertEqual(n_long_states(10, 10, 3), 12 * 100 * 97)

    def test_every_state_is_reachable(self):
        for size, length in (((4, 4), 2), ((4, 4), 3)):
            game = ShortSnake(size, seed=0, length=length)
            states = set(all_long_states(size, length))
            self.assertEqual(len(states), n_long_states(*size, length))
            self.assertEqual(reachable_states(game.save_state(), size), states)

    def test_model_matches_game(self):
        rng = random.Random(1)
        for size, length in (((5, 6), 3), ((6, 6), 5)):
            game = ShortSnake(size, seed=2, length=length)
            states = all_long_states(size, length)
            for _ in range(1500):
                state = rng.choice(states)
                action = rng.choice(MOVE_ARGUMENTS)
                game.restore_state(state)
                reward = game.move(action)
                outcomes = {(s, r) for _, s, r in transitions(state, action, size)}
                self.assertIn((game.save_state(), reward), outcomes)


class TestGreedyPlayerUnchanged(unittest.TestCase):
    def test_player_with_long_snake(self):
        game = ShortSnake(10, seed=0, length=5)
        player = GreedyPlayer(game, 0.1, seed=1)
        rewards = play(player, game, 2000)
        self.assertEqual(int(rewards.sum()), game.score)
        self.assertTrue(all(len(state) == 4 for state, _ in player.action_values))
        self.assertTrue(all(a in actions(s) for s, a in player.action_values))

    def test_long_snake_learns(self):
        learned = run(0.1, 100_000, seed=0, length=2)[-10_000:].mean()
        random_play = run(0.1, 100_000, seed=0, eps=1.0, length=2)[-10_000:].mean()
        self.assertGreater(learned, random_play + 0.3)


if __name__ == "__main__":
    unittest.main()
