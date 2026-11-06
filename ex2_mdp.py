"""Exercice 2 : réponses calculées à partir du modèle MDP et figure du MDP partiel.

    python ex2_mdp.py    # affiche les réponses et crée rapport/figures/ex2_mdp.png
"""

import pathlib
import sys
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

from mdp import (  # noqa: E402
    MOVE_ARGUMENTS, actions, all_states, n_states, reachable_states, state_of,
    transitions,
)
from snake import DOWN, LEFT, RIGHT, UP, ShortSnake  # noqa: E402

FIGURE = pathlib.Path(__file__).resolve().parent / "rapport" / "figures" / "ex2_mdp.png"
ARROW = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


def print_answers():
    game = ShortSnake(10, seed=0)
    state = state_of(game)

    print("Q2.1  état = (tête, direction, fruit) =", state)

    reached = reachable_states(state, 10)
    print(f"Q2.2  grille 10 x 10 : {n_states(10, 10)} états "
          f"(énumérés : {len(all_states(10))}, atteignables : {len(reached)})")

    print("Q2.3  |S| = 4·N·M·(N·M − 1) :",
          {f"{n}x{m}": n_states(n, m) for n, m in [(3, 3), (4, 5), (10, 10), (20, 20)]})

    print("Q2.4  arguments de move :", MOVE_ARGUMENTS)
    effects = {}
    for action in MOVE_ARGUMENTS:
        key = tuple(transitions(state, action, 10))
        effects.setdefault(key, []).append(action)
    print("      actions regroupées par effet dans", state, ":", list(effects.values()))

    for direction in (UP, DOWN, LEFT, RIGHT):
        print(f"Q2.5  A(s) si direction = {direction:5} :", actions(((5, 5), direction, (0, 0))))

    rewards = sorted({
        r for s in all_states(10) for a in actions(s) for _, _, r in transitions(s, a, 10)
    })
    print("Q2.6  récompenses possibles :", rewards)
    print("      exemple bord  :", transitions(((5, 9), RIGHT, (0, 0)), RIGHT, 10))
    eat = transitions(((1, 0), UP, (0, 0)), UP, 10)
    print(f"      exemple fruit : {len(eat)} états suivants, p = 1/{len(eat)}, r = {eat[0][2]}")


# --- Q2.7 : représentation graphique ---------------------------------------

CELL = 0.4


def draw_state(ax, center, state, size, name, label="below"):
    """Dessine un état comme une mini-grille : tête verte fléchée, fruit rouge."""
    n_rows, n_cols = size
    x0 = center[0] - n_cols * CELL / 2
    y0 = center[1] - n_rows * CELL / 2
    (h_row, h_col), direction, (f_row, f_col) = state
    for row in range(n_rows):
        for col in range(n_cols):
            x, y = x0 + col * CELL, y0 + (n_rows - 1 - row) * CELL
            ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor="white",
                                   edgecolor="#9aa5b1", linewidth=0.8))
            if (row, col) == (h_row, h_col):
                ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor="#2e8b57"))
                ax.text(x + CELL / 2, y + CELL / 2, ARROW[direction], color="white",
                        ha="center", va="center", fontsize=11, fontweight="bold")
            if (row, col) == (f_row, f_col):
                ax.add_patch(Circle((x + CELL / 2, y + CELL / 2), CELL * 0.3,
                                    facecolor="#d62728"))
    ax.add_patch(Rectangle((x0, y0), n_cols * CELL, n_rows * CELL, fill=False,
                           edgecolor="#1d3557", linewidth=1.6))
    text = f"{name}\n{state[0]}, {state[1]}, {state[2]}"
    if label == "above":
        ax.text(center[0], y0 + n_rows * CELL + 0.12, text, ha="center", va="bottom", fontsize=8)
    elif label == "left":
        ax.text(x0 - 0.15, center[1], text, ha="right", va="center", fontsize=8)
    else:
        ax.text(center[0], y0 - 0.12, text, ha="center", va="top", fontsize=8)


def edge_point(center, toward, half):
    """Point du bord d'une boîte carrée (demi-côté `half`) en direction de `toward`."""
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    t = half / max(abs(dx), abs(dy))
    return (center[0] + dx * t, center[1] + dy * t)


def arrow(ax, start, end, color="#333333", rad=0.0, width=1.3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 connectionstyle=f"arc3,rad={rad}", color=color,
                                 linewidth=width, shrinkA=0, shrinkB=0))


def action_dot(ax, pos, name, offset=(0, 0.22)):
    ax.add_patch(Circle(pos, 0.09, facecolor="black", zorder=5))
    ax.text(pos[0] + offset[0], pos[1] + offset[1], name, ha="center", va="center",
            fontsize=8.5, fontweight="bold")


def proba_label(prob, reward):
    frac = Fraction(prob).limit_denominator(1000)
    return f"{frac}, {reward:+d}" if reward else f"{frac}, 0"


REWARD_COLOR = {-10: "#c0392b", 0: "#333333", 10: "#1e8449"}


def draw_mdp_figure(path):
    size = (3, 3)
    half = 3 * CELL / 2 + 0.03

    # États du cycle, obtenus avec le modèle p(s', r | s, a)
    A = ((1, 0), RIGHT, (0, 0))
    [(p_ab, B, r_ab)] = transitions(A, RIGHT, size)
    [(p_bc, C, r_bc)] = transitions(B, RIGHT, size)
    [(p_ca, back, r_ca)] = transitions(C, RIGHT, size)
    assert back == A, "RIGHT trois fois depuis A doit revenir en A"
    [(p_ad, D, r_ad)] = transitions(A, DOWN, size)
    eaten = transitions(A, UP, size)  # fruit mangé : 8 états suivants

    pos = {"A": (5.0, 4.6), "B": (9.5, 4.6), "C": (14.0, 4.6), "D": (2.6, 1.2)}
    top = [(2.6, 8.4), (5.0, 8.4), (7.4, 8.4)]

    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0.6, 16.6)
    ax.set_ylim(-0.2, 9.9)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_state(ax, pos["A"], A, size, "A", label="left")
    draw_state(ax, pos["B"], B, size, "B", label="above")
    draw_state(ax, pos["C"], C, size, "C", label="above")
    draw_state(ax, pos["D"], D, size, "D")
    for i, (prob, state, reward) in enumerate(eaten[:3]):
        draw_state(ax, top[i], state, size, f"E{i + 1}", label="above")

    cycle = "#1d4e89"
    # A --RIGHT--> B --RIGHT--> C
    for src, dst, dot, prob, reward in (("A", "B", (7.25, 4.6), p_ab, r_ab),
                                        ("B", "C", (11.75, 4.6), p_bc, r_bc)):
        arrow(ax, edge_point(pos[src], dot, half), (dot[0] - 0.1, dot[1]), cycle, width=2)
        action_dot(ax, dot, "RIGHT")
        arrow(ax, (dot[0] + 0.1, dot[1]), edge_point(pos[dst], dot, half), cycle, width=2)
        ax.text(dot[0] + 0.55, dot[1] - 0.25, proba_label(prob, reward),
                fontsize=8.5, color=REWARD_COLOR[reward], ha="center")

    # C --RIGHT--> A (traversée du bord droit : -10)
    dot_c = (15.7, 4.6)
    arrow(ax, edge_point(pos["C"], dot_c, half), (dot_c[0] - 0.1, dot_c[1]), cycle, width=2)
    action_dot(ax, dot_c, "RIGHT")
    arrow(ax, (dot_c[0], dot_c[1] - 0.1), (pos["A"][0] + 0.2, pos["A"][1] - half),
          cycle, rad=-0.4, width=2)
    ax.text(9.9, 1.05, proba_label(p_ca, r_ca) + "  (bord droit franchi)",
            fontsize=9, color=REWARD_COLOR[r_ca], ha="center")

    # A --DOWN--> D
    dot_d = (3.8, 2.95)
    arrow(ax, edge_point(pos["A"], dot_d, half), dot_d)
    action_dot(ax, dot_d, "DOWN", offset=(-0.6, 0.12))
    arrow(ax, dot_d, edge_point(pos["D"], dot_d, half))
    ax.text(3.45, 2.1, proba_label(p_ad, r_ad), fontsize=8.5, ha="left")

    # A --UP--> fruit mangé, nouveau fruit aléatoire
    dot_u = (5.0, 6.35)
    arrow(ax, edge_point(pos["A"], dot_u, half), dot_u)
    action_dot(ax, dot_u, "UP", offset=(0.35, -0.05))
    label_spots = [((3.75, 7.15), "right"), ((5.12, 7.55), "left"), ((6.2, 6.85), "left")]
    for i, (prob, _, reward) in enumerate(eaten[:3]):
        arrow(ax, dot_u, edge_point(top[i], dot_u, half), REWARD_COLOR[reward])
        (x, y), align = label_spots[i]
        ax.text(x, y, proba_label(prob, reward), fontsize=8,
                color=REWARD_COLOR[reward], ha=align, va="center")
    ax.text(8.6, 8.4, f"… + {len(eaten) - 3} autres états\n(une position par\nnouveau fruit)",
            fontsize=8.5, va="center", color="#555")

    ax.text(1.0, 5.6, "LEFT : interdit (demi-tour)\nNone ≡ RIGHT", fontsize=8.5,
            color="#777", style="italic", va="center")
    ax.text(16.5, 9.7, "Cycle A → B → C → A en bleu\nGrille 3 × 3, état = (tête, direction, fruit)",
            fontsize=8.5, ha="right", va="top", color="#333")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Q2.7  figure :", path)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):   # accents lisibles même si la sortie est redirigée
        if hasattr(stream, "reconfigure"):    # absent sous IDLE, None sans console (pythonw)
            stream.reconfigure(encoding="utf-8")
    print_answers()
    draw_mdp_figure(FIGURE)
