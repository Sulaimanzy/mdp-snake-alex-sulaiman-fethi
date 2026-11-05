"""Exercice 2 : modélisation de ShortSnake comme un processus de décision markovien fini.

Un état est le tuple (position de la tête, direction, position du fruit). La
dynamique p(s', r | s, a) est donnée par `transitions`.
"""

from collections import deque

from snake import (
    DIRECTIONS, DOWN, FRUIT_REWARD, LEFT, OPPOSITE, RIGHT, UP, WALL_PENALTY,
)

# Arguments acceptés par ShortSnake.move
MOVE_ARGUMENTS = [None, UP, DOWN, LEFT, RIGHT]


def _size(size):
    return (size, size) if isinstance(size, int) else tuple(size)


def cells(size):
    n_rows, n_cols = _size(size)
    return [(row, col) for row in range(n_rows) for col in range(n_cols)]


# Q2.1 -- les informations qui constituent un état
def state_of(game):
    """État du MDP pour une partie en cours : (tête, direction, fruit)."""
    return (game.snake_pos, game.snake_dir, game.fruit_pos)


# Q2.2 / Q2.3 -- nombre d'états
def n_states(n_rows, n_cols):
    """|S| = N·M positions de tête × 4 directions × (N·M − 1) positions du fruit."""
    n_cells = n_rows * n_cols
    return n_cells * len(DIRECTIONS) * (n_cells - 1)


def all_states(size):
    """Énumère tous les états : le fruit n'est jamais sous la tête."""
    grid_cells = cells(size)
    return [
        (head, direction, fruit)
        for head in grid_cells
        for direction in DIRECTIONS
        for fruit in grid_cells
        if fruit != head
    ]


# Q2.4 / Q2.5 -- actions disponibles dans un état
def actions(state):
    """A(s) : les quatre directions sauf le demi-tour (None équivaut à la direction courante)."""
    direction = state[1]
    return [d for d in DIRECTIONS if d != OPPOSITE[direction]]


# Q2.6 -- dynamique et récompenses
def transitions(state, action, size):
    """p(s', r | s, a) sous forme de liste de (probabilité, état suivant, récompense)."""
    n_rows, n_cols = _size(size)
    (row, col), direction, fruit = state

    if action is not None and action != OPPOSITE[direction]:
        direction = action
    d_row, d_col = DIRECTIONS[direction]
    row, col = row + d_row, col + d_col

    reward = 0
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        row, col = row % n_rows, col % n_cols
        reward += WALL_PENALTY
    head = (row, col)

    if head != fruit:  # déplacement déterministe
        return [(1.0, (head, direction, fruit), reward)]

    # Fruit mangé : le nouveau fruit est uniforme sur les N·M − 1 autres cases
    reward += FRUIT_REWARD
    free_cells = [cell for cell in cells(size) if cell != head]
    prob = 1 / len(free_cells)
    return [(prob, (head, direction, new_fruit), reward) for new_fruit in free_cells]


def reachable_states(start, size):
    """États atteignables depuis `start` (parcours en largeur du graphe du MDP)."""
    seen = {start}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for action in actions(state):
            for _, next_state, _ in transitions(state, action, size):
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append(next_state)
    return seen
