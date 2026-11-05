"""Tests de l'exercice 1 : mécanique de base de ShortSnake.

Lancement depuis la racine du dépôt : py -m unittest discover tests
"""

import unittest

from snake import (
    DOWN, EMPTY, FRUIT, LEFT, RIGHT, SNAKE, UP, ShortSnake,
)


def put_fruit(game, pos):
    """Déplace le fruit sur `pos` en gardant la grille cohérente."""
    game._set_cell(game.fruit_pos, EMPTY)
    game.fruit_pos = pos
    game._set_cell(pos, FRUIT)


def count_cells(game, value):
    return sum(row.count(value) for row in game.grid)


class TestInit(unittest.TestCase):
    def test_square_grid(self):
        game = ShortSnake(10, seed=0)
        self.assertEqual(len(game.grid), 10)
        self.assertTrue(all(len(row) == 10 for row in game.grid))

    def test_rectangular_grid(self):
        game = ShortSnake((4, 7), seed=0)
        self.assertEqual((game.n_rows, game.n_cols), (4, 7))
        self.assertEqual(game.snake_pos, (2, 3))

    def test_initial_state(self):
        game = ShortSnake(10, seed=0)
        self.assertEqual(game.snake_pos, (5, 5))
        self.assertEqual(game.snake_dir, RIGHT)
        self.assertEqual(game.score, 0)
        self.assertNotEqual(game.fruit_pos, game.snake_pos)
        self.assertEqual(count_cells(game, SNAKE), 1)
        self.assertEqual(count_cells(game, FRUIT), 1)

    def test_fruit_is_random(self):
        positions = {ShortSnake(10, seed=s).fruit_pos for s in range(50)}
        self.assertGreater(len(positions), 10)

    def test_too_small_grid(self):
        with self.assertRaises(ValueError):
            ShortSnake(1)


class TestMove(unittest.TestCase):
    def setUp(self):
        self.game = ShortSnake(10, seed=0)
        put_fruit(self.game, (0, 0))  # fruit loin du serpent

    def test_none_keeps_direction(self):
        reward = self.game.move(None)
        self.assertEqual(reward, 0)
        self.assertEqual(self.game.snake_pos, (5, 6))
        self.assertEqual(self.game.snake_dir, RIGHT)

    def test_turn(self):
        self.game.move(UP)
        self.assertEqual(self.game.snake_pos, (4, 5))
        self.assertEqual(self.game.snake_dir, UP)

    def test_no_u_turn(self):
        self.game.move(LEFT)  # demi-tour interdit : on continue à droite
        self.assertEqual(self.game.snake_pos, (5, 6))
        self.assertEqual(self.game.snake_dir, RIGHT)

    def test_grid_follows_snake(self):
        self.game.move(DOWN)
        self.assertEqual(self.game.grid[5][5], EMPTY)
        self.assertEqual(self.game.grid[6][5], SNAKE)
        self.assertEqual(count_cells(self.game, SNAKE), 1)

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            self.game.move("DIAGONAL")

    def test_wrap_right_edge(self):
        for _ in range(4):  # (5,5) -> (5,9) : 4 pas sans pénalité
            self.assertEqual(self.game.move(), 0)
        self.assertEqual(self.game.move(), -10)
        self.assertEqual(self.game.snake_pos, (5, 0))
        self.assertEqual(self.game.score, -10)

    def test_wrap_top_edge(self):
        for _ in range(5):
            self.game.move(UP)
        self.assertEqual(self.game.snake_pos, (0, 5))
        self.assertEqual(self.game.move(UP), -10)
        self.assertEqual(self.game.snake_pos, (9, 5))

    def test_wrap_bottom_edge(self):
        for _ in range(4):
            self.game.move(DOWN)
        self.assertEqual(self.game.snake_pos, (9, 5))
        self.assertEqual(self.game.move(DOWN), -10)
        self.assertEqual(self.game.snake_pos, (0, 5))

    def test_wrap_left_edge(self):
        self.game.move(UP)
        for _ in range(5):
            self.game.move(LEFT)
        self.assertEqual(self.game.snake_pos, (4, 0))
        self.assertEqual(self.game.move(LEFT), -10)
        self.assertEqual(self.game.snake_pos, (4, 9))

    def test_eat_fruit(self):
        put_fruit(self.game, (5, 6))
        self.assertEqual(self.game.move(), 10)
        self.assertEqual(self.game.score, 10)
        self.assertNotEqual(self.game.fruit_pos, (5, 6))
        self.assertNotEqual(self.game.fruit_pos, self.game.snake_pos)
        self.assertEqual(count_cells(self.game, FRUIT), 1)

    def test_eat_fruit_after_wrap(self):
        self.game.snake_pos, self.game.grid[5][5] = (5, 9), EMPTY
        self.game._set_cell((5, 9), SNAKE)
        put_fruit(self.game, (5, 0))
        self.assertEqual(self.game.move(), 0)  # -10 (bord) + 10 (fruit)


class TestStr(unittest.TestCase):
    def test_str(self):
        game = ShortSnake((3, 4), seed=0)
        text = str(game)
        lines = text.splitlines()
        self.assertEqual(lines[0], "+----+")
        self.assertEqual(len(lines), 3 + 2 + 1)
        self.assertIn(">", text)
        self.assertIn("*", text)
        self.assertEqual(lines[-1], "Score : 0")


if __name__ == "__main__":
    unittest.main()
