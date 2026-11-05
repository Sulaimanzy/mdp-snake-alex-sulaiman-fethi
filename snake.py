"""TP MDP : Snake -- variante « courte » du jeu Snake.

Le serpent ne grandit pas et ne meurt pas : il ne fait que gagner ou perdre
des points. C'est une tâche continue (pas d'état final).
"""

import operator
import random

# Contenu possible d'une case de la grille
EMPTY, SNAKE, FRUIT = 0, 1, 2

# Directions : (déplacement en ligne, déplacement en colonne)
UP, DOWN, LEFT, RIGHT = "UP", "DOWN", "LEFT", "RIGHT"
DIRECTIONS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Récompenses
WALL_PENALTY = -10
FRUIT_REWARD = 10

# Affichage
HEAD_SYMBOL = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}
CELL_SYMBOL = {EMPTY: ".", FRUIT: "*"}


class ShortSnake:
    """Mécanique du jeu Snake dans sa variante courte.

    Les coordonnées sont des tuples (ligne, colonne), (0, 0) étant le coin
    en haut à gauche.
    """

    def __init__(self, size=10, seed=None):
        # Q1.1 -- taille de la grille : un entier (grille carrée) ou (lignes, colonnes)
        if isinstance(size, int):
            size = (size, size)
        self.n_rows, self.n_cols = size
        if self.n_rows < 1 or self.n_cols < 1 or self.n_rows * self.n_cols < 2:
            raise ValueError("la grille doit contenir au moins 2 cases")
        self.grid = [[EMPTY] * self.n_cols for _ in range(self.n_rows)]
        self.rng = random.Random(seed)

        # Q1.2 -- position et direction initiales : au milieu, vers la droite
        self.snake_pos = (self.n_rows // 2, self.n_cols // 2)
        self.snake_dir = RIGHT
        self._set_cell(self.snake_pos, SNAKE)

        # Q1.3 -- fruit placé aléatoirement (jamais sous le serpent)
        self.fruit_pos = self._place_fruit()

        # Q1.4 -- score initial
        self.score = 0

    def _set_cell(self, pos, value):
        row, col = pos
        self.grid[row][col] = value

    def _place_fruit(self):
        """Pose un fruit sur une case libre tirée au hasard et renvoie sa position."""
        free_cells = [
            (row, col)
            for row in range(self.n_rows)
            for col in range(self.n_cols)
            if self.grid[row][col] == EMPTY
        ]
        pos = self.rng.choice(free_cells)
        self._set_cell(pos, FRUIT)
        return pos

    def move(self, direction=None):
        """Fait avancer le serpent d'une case et renvoie la récompense obtenue.

        `direction` est UP, DOWN, LEFT, RIGHT, ou None pour continuer tout droit.
        """
        # Q1.5 -- None : on garde la direction ; demi-tour interdit (ignoré)
        if direction is not None:
            if direction not in DIRECTIONS:
                raise ValueError(f"direction inconnue : {direction!r}")
            if direction != OPPOSITE[self.snake_dir]:
                self.snake_dir = direction

        d_row, d_col = DIRECTIONS[self.snake_dir]
        row, col = self.snake_pos[0] + d_row, self.snake_pos[1] + d_col
        reward = 0

        # Q1.6 -- bord atteint : on réapparaît de l'autre côté et on perd 10 points
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            row %= self.n_rows
            col %= self.n_cols
            reward += WALL_PENALTY

        # Déplacement de la tête
        self._set_cell(self.snake_pos, EMPTY)
        self.snake_pos = (row, col)
        self._set_cell(self.snake_pos, SNAKE)

        # Q1.7 -- fruit mangé : +10 points et nouveau fruit aléatoire
        if self.snake_pos == self.fruit_pos:
            reward += FRUIT_REWARD
            self.fruit_pos = self._place_fruit()

        self.score += reward
        return reward

    # Q3.1 -- sauvegarde de l'état du jeu
    def save_state(self):
        """Renvoie l'état actuel du jeu : (tête, direction, fruit).

        C'est l'état du MDP (exercice 2) : le score n'en fait pas partie, c'est la
        somme des récompenses déjà reçues.
        """
        return (self.snake_pos, self.snake_dir, self.fruit_pos)

    # Q3.2 -- restauration d'un état sauvegardé
    def _as_cell(self, pos):
        """Convertit `pos` en tuple (ligne, colonne) d'entiers dans la grille."""
        try:
            row, col = (operator.index(x) for x in pos)
        except (TypeError, ValueError):
            raise ValueError(f"case invalide : {pos!r}") from None
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise ValueError(f"case hors de la grille : {(row, col)}")
        return (row, col)

    def restore_state(self, state):
        """Remet le jeu dans un état renvoyé par save_state (le score est inchangé).

        Toutes les vérifications ont lieu avant la moindre modification du jeu.
        """
        try:
            head, direction, fruit = state
        except (TypeError, ValueError):
            raise ValueError(f"état invalide : {state!r}") from None
        head, fruit = self._as_cell(head), self._as_cell(fruit)
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            raise ValueError(f"direction inconnue : {direction!r}")
        if head == fruit:
            raise ValueError("le fruit ne peut pas être sous le serpent")

        self.grid = [[EMPTY] * self.n_cols for _ in range(self.n_rows)]
        self.snake_pos, self.snake_dir, self.fruit_pos = head, direction, fruit
        self._set_cell(head, SNAKE)
        self._set_cell(fruit, FRUIT)

    # Q1.8 -- affichage de la grille et du score
    def __str__(self):
        border = "+" + "-" * self.n_cols + "+"
        lines = [border]
        for row in self.grid:
            cells = (
                HEAD_SYMBOL[self.snake_dir] if cell == SNAKE else CELL_SYMBOL[cell]
                for cell in row
            )
            lines.append("|" + "".join(cells) + "|")
        lines.append(border)
        lines.append(f"Score : {self.score}")
        return "\n".join(lines)
