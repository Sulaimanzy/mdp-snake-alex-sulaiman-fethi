"""Tests de l'exercice 2 : le modèle MDP est cohérent avec le jeu.

Lancement depuis la racine du dépôt : py -m unittest discover tests
"""

import random
import unittest

from mdp import (
    MOVE_ARGUMENTS, actions, all_states, n_states, reachable_states, state_of,
    transitions,
)
from snake import DOWN, LEFT, RIGHT, UP, ShortSnake


class TestStates(unittest.TestCase):
    def test_state_of_initial_game(self):
        game = ShortSnake(10, seed=0)
        self.assertEqual(state_of(game), ((5, 5), RIGHT, game.fruit_pos))

    def test_count_10x10(self):
        self.assertEqual(n_states(10, 10), 39600)
        self.assertEqual(len(all_states(10)), 39600)

    def test_count_formula(self):
        for n_rows, n_cols in [(2, 2), (3, 3), (3, 4), (5, 2)]:
            states = all_states((n_rows, n_cols))
            self.assertEqual(len(states), n_states(n_rows, n_cols))
            self.assertEqual(len(set(states)), len(states))

    def test_every_state_is_reachable(self):
        for size in [(3, 3), (3, 4)]:
            game = ShortSnake(size, seed=0)
            reached = reachable_states(state_of(game), size)
            self.assertEqual(reached, set(all_states(size)))


class TestActions(unittest.TestCase):
    def test_no_u_turn(self):
        self.assertEqual(actions(((1, 1), RIGHT, (0, 0))), [UP, DOWN, RIGHT])
        self.assertEqual(actions(((1, 1), UP, (0, 0))), [UP, LEFT, RIGHT])

    def test_u_turn_and_none_are_redundant(self):
        state = ((1, 1), RIGHT, (0, 0))
        same = transitions(state, RIGHT, 3)
        self.assertEqual(transitions(state, None, 3), same)
        self.assertEqual(transitions(state, LEFT, 3), same)


class TestTransitions(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for state in all_states(3):
            for action in MOVE_ARGUMENTS:
                total = sum(p for p, _, _ in transitions(state, action, 3))
                self.assertAlmostEqual(total, 1.0)

    def test_rewards(self):
        rewards = {
            r
            for state in all_states(3)
            for action in actions(state)
            for _, _, r in transitions(state, action, 3)
        }
        self.assertEqual(rewards, {-10, 0, 10})

    def test_fruit_is_stochastic(self):
        result = transitions(((1, 0), RIGHT, (0, 0)), UP, 3)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(p == 1 / 8 and r == 10 for p, _, r in result))

    def test_model_matches_game(self):
        rng = random.Random(1)
        size = (4, 5)
        game = ShortSnake(size, seed=2)
        states = all_states(size)
        for _ in range(2000):
            state = rng.choice(states)
            action = rng.choice(MOVE_ARGUMENTS)
            game.restore_state(state)
            reward = game.move(action)
            outcomes = {(s, r) for _, s, r in transitions(state, action, size)}
            self.assertIn((state_of(game), reward), outcomes)


if __name__ == "__main__":
    unittest.main()
