"""Exercice 2 : modélisation de ShortSnake comme un processus de décision markovien fini.

Un état est le tuple (position de la tête, direction, position du fruit). La
dynamique p(s', r | s, a) est donnée par `transitions`.
"""

from collections import deque

from snake import (
    DIRECTIONS, DOWN, FRUIT_REWARD, LEFT, OPPOSITE, RIGHT, SELF_PENALTY, UP, WALL_PENALTY,
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
    """État du MDP pour une partie en cours : (tête, direction, fruit), plus le corps pour
    un serpent de plusieurs cases."""
    if getattr(game, "body", None):   # serpent de plusieurs cases (exercice 4)
        return game.save_state()
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
_LEGAL_ACTIONS = {
    direction: [d for d in DIRECTIONS if d != OPPOSITE[direction]]
    for direction in DIRECTIONS
}


def actions(state):
    """A(s) : les quatre directions sauf le demi-tour (None équivaut à la direction courante).

    La liste renvoyée est partagée entre les appels : ne pas la modifier.
    """
    return _LEGAL_ACTIONS[state[1]]


# Q2.6 -- dynamique et récompenses
def transitions(state, action, size):
    """p(s', r | s, a) sous forme de liste de (probabilité, état suivant, récompense)."""
    if len(state) == 4:   # serpent de longueur ≥ 2 (exercice 4)
        return long_transitions(state, action, size)
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


# --- Exercice 4 : serpent de longueur L ≥ 2 ------------------------------------

def long_transitions(state, action, size):
    """p(s', r | s, a) pour l'état (tête, direction, fruit, corps) d'un serpent de longueur ≥ 2."""
    n_rows, n_cols = _size(size)
    head, direction, fruit, body = state

    if action is not None and action != OPPOSITE[direction]:
        direction = action
    d_row, d_col = DIRECTIONS[direction]
    row, col = head[0] + d_row, head[1] + d_col

    reward = 0
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        row, col = row % n_rows, col % n_cols
        reward += WALL_PENALTY
    new_head = (row, col)
    new_body = (head,) + body[:-1]          # le corps suit la tête, la queue avance
    if new_head in new_body:                # auto-intersection
        reward += SELF_PENALTY

    if new_head != fruit:
        return [(1.0, (new_head, direction, fruit, new_body), reward)]
    reward += FRUIT_REWARD
    occupied = {new_head, *new_body}
    free_cells = [cell for cell in cells(size) if cell not in occupied]
    prob = 1 / len(free_cells)
    return [(prob, (new_head, direction, new_fruit, new_body), reward) for new_fruit in free_cells]


def body_shapes(length):
    """Formes du serpent : directions des déplacements, de la tête vers la queue, sans
    demi-tour entre deux déplacements successifs. La première est la direction courante."""
    shapes = [(d,) for d in DIRECTIONS]
    for _ in range(length - 2):
        shapes = [s + (d,) for s in shapes for d in DIRECTIONS if d != OPPOSITE[s[-1]]]
    return shapes


def snake_cells(head, shape, length, size):
    """Cases du serpent (tête d'abord) pour une forme donnée."""
    n_rows, n_cols = _size(size)
    snake = [head]
    for direction in shape[:length - 1]:
        d_row, d_col = DIRECTIONS[direction]
        row, col = snake[-1]
        snake.append(((row - d_row) % n_rows, (col - d_col) % n_cols))
    return snake


def all_long_states(size, length):
    """Énumère les états d'un serpent de longueur `length` (format de save_state)."""
    states = []
    for head in cells(size):
        for shape in body_shapes(length):
            snake = snake_cells(head, shape, length, size)
            occupied = set(snake)
            for fruit in cells(size):
                if fruit in occupied:
                    continue
                if length == 1:
                    states.append((head, shape[0], fruit))
                else:
                    states.append((head, shape[0], fruit, tuple(snake[1:])))
    return states


# Q4.2 -- nombre d'états pour un serpent de longueur L
def n_long_states(n_rows, n_cols, length):
    """|S| = N·M positions de tête × somme, sur les formes du corps, des cases libres
    pour le fruit (N·M − nombre de cases occupées par le serpent).

    Valable quand chaque côté de la grille fait au moins `length` cases (condition imposée
    par ShortSnake) : deux formes différentes donnent alors deux serpents différents."""
    n_cells = n_rows * n_cols
    head = (0, 0)   # le décompte ne dépend pas de la position de la tête (grille torique)
    return n_cells * sum(
        n_cells - len(set(snake_cells(head, shape, length, (n_rows, n_cols))))
        for shape in body_shapes(length)
    )


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
