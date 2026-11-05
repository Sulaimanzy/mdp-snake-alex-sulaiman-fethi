"""Exercice 4 (bonus) : serpent plus long.

    py ex4_long_snake.py demo   # Q4.3 : un serpent de 5 cases se mord la queue
    py ex4_long_snake.py q2     # Q4.1-Q4.2 : nombre d'états (formule, énumération, accessibilité)
    py ex4_long_snake.py q5     # Q4.5 : graphiques de l'exercice 3 pour un serpent de 2 cases
    py ex4_long_snake.py q6     # Q4.6 : vitesse d'apprentissage, 1 case contre 2 cases
    py ex4_long_snake.py q7     # Q4.7 : serpents de 1 à 8 cases, entraînements longs
    py ex4_long_snake.py all    # tout (≈ 9 min sur 32 cœurs, ≈ 7 Go de RAM)

Les figures vont dans rapport/figures/, les résultats chiffrés dans resultats/.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import ex3_qlearning as ex3
from ex3_qlearning import ALPHA, EPS, INK, RANDOM_COLOR, SIZE, mean_ci, new_figure, save_csv, \
    save_figure, style
from greedy_player import GreedyPlayer
from mdp import all_long_states, body_shapes, n_long_states, reachable_states
from snake import DOWN, LEFT, RIGHT, UP, ShortSnake

LENGTHS = [1, 2, 3, 4, 5, 6, 8]
LENGTH_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4",
                 6: "#008300", 8: "#6250d6"}


# --- Q4.1 / Q4.2 : nombre d'états ----------------------------------------------

def q2(pool):
    n_cells = SIZE * SIZE
    print("Q4.2 : nombre d'états")
    print(f"  serpent de 2 cases, grille 10 x 10 : 4·N·M·(N·M − 2) = {4 * n_cells * (n_cells - 2)}"
          f" ; comptage par formes (n_long_states) : {n_long_states(SIZE, SIZE, 2)}")

    # Contrôle indépendant : les états atteints en jouant (parcours en largeur du modèle,
    # lui-même vérifié contre le jeu par les tests) doivent être exactement ceux comptés.
    checks = []
    for size, length in (((4, 4), 2), ((5, 5), 2), ((4, 4), 3), ((4, 5), 4), ((5, 5), 5)):
        game = ShortSnake(size, seed=0, length=length)
        states = set(all_long_states(size, length))
        reached = reachable_states(game.save_state(), size)
        checks.append([f"{size[0]}x{size[1]}", length, n_long_states(*size, length), len(states),
                       len(reached), reached == states])
        print(f"  grille {size[0]} x {size[1]}, {length} cases : comptage {checks[-1][2]}, "
              f"états distincts énumérés {len(states)}, atteignables depuis l'état initial "
              f"{len(reached)}")
    save_csv("ex4_accessibilite.csv",
             ["grille", "longueur", "comptage", "enumeration", "atteignables", "identiques"], checks)

    rows = []
    for length in LENGTHS:
        states = n_long_states(SIZE, SIZE, length)
        rows.append([length, len(body_shapes(length)), states, 3 * states])
        print(f"  {length} case(s) : {len(body_shapes(length)):5} formes, {states:9} états, "
              f"{3 * states:9} paires (état, action)")
    save_csv("ex4_etats.csv", ["longueur", "formes", "etats", "paires_etat_action"], rows)


# --- Q4.5 : les graphiques de l'exercice 3 pour un serpent de 2 cases ----------

def q5(pool):
    ex3.q10(pool, length=2, prefix="ex4")
    ex3.q11(pool, length=2, prefix="ex4")
    ex3.q12(pool, length=2, prefix="ex4")
    rows_1000 = ex3.q13(pool, length=2, prefix="ex4")
    ex3.q14(pool, rows_1000, length=2, prefix="ex4")
    ex3.long_horizon(pool, length=2, prefix="ex4")


# --- Q4.6 : vitesse d'accroissement de la récompense, 1 case contre 2 ----------

def first_step_above(curve, block, threshold):
    """Première étape (fin de bloc) où la récompense moyenne atteint `threshold`."""
    above = np.nonzero(curve >= threshold)[0]
    return int((above[0] + 1) * block) if len(above) else None


def q6(pool):
    print("Q4.6 : vitesse d'apprentissage, serpent de 1 case contre 2 cases")
    n_runs = ex3.N_RUNS_LONG
    short = {length: ex3.run_many(pool, 0.5, 10_000, n_runs, length=length) for length in (1, 2)}
    rows, gains = [], {}
    for length, rewards in short.items():
        first, ci_first = mean_ci(rewards[:, :1000].mean(axis=1))
        last, ci_last = mean_ci(rewards[:, -1000:].mean(axis=1))
        gains[length] = rewards[:, -1000:].mean(axis=1) - rewards[:, :1000].mean(axis=1)
        gain, ci_gain = mean_ci(gains[length])
        rows.append([length, "court", 0.5, 10_000, n_runs, round(float(first), 4),
                     round(float(ci_first), 4), round(float(last), 4), round(float(ci_last), 4),
                     round(float(gain), 4), round(float(ci_gain), 4), "", ""])
        print(f"  {length} case(s), γ = 0.5, 10 000 étapes : {first:+.3f} ± {ci_first:.3f} sur les "
              f"1000 premières, {last:+.3f} ± {ci_last:.3f} sur les 1000 dernières "
              f"(gain {gain:+.3f} ± {ci_gain:.3f})")
    diff, ci_diff = mean_ci(gains[2] - gains[1])   # mêmes graines : écart apparié
    rows.append(["2 moins 1", "court", 0.5, 10_000, n_runs, "", "", "", "", round(float(diff), 4),
                 round(float(ci_diff), 4), "", ""])
    print(f"  écart des gains (2 cases moins 1 case, apparié) : {diff:+.3f} ± {ci_diff:.3f} "
          f"({ex3.verdict(diff, ci_diff)})")

    curves = {}
    for length in (1, 2):
        for gamma in (0.1, 0.5):
            tasks = [(gamma, ex3.LONG_STEPS, seed, ex3.LONG_BLOCK, length)
                     for seed in range(ex3.LONG_RUNS)]
            blocks = np.array(list(pool.map(ex3._run_blocks, tasks)))
            curves[length, gamma] = blocks.mean(axis=0)
            last, ci_last = mean_ci(blocks[:, -10:].mean(axis=1))
            zero = first_step_above(curves[length, gamma], ex3.LONG_BLOCK, 0)
            half = first_step_above(curves[length, gamma], ex3.LONG_BLOCK, 0.5)
            rows.append([length, "long", gamma, ex3.LONG_STEPS, ex3.LONG_RUNS, "", "",
                         round(float(last), 4), round(float(ci_last), 4), "", "", zero, half])
            print(f"  {length} case(s), γ = {gamma}, 1e6 étapes : {last:+.3f} ± {ci_last:.3f} sur les "
                  f"1e5 dernières ; récompense ≥ 0 à l'étape {zero}, ≥ 0.5 à l'étape {half}")
    save_csv("ex4_q6_comparaison.csv",
             ["longueur", "horizon", "gamma", "etapes", "relances", "debut", "ic95_debut", "fin",
              "ic95_fin", "gain", "ic95_gain", "etape_recompense_0", "etape_recompense_0.5"], rows)

    fig, (ax1, ax2) = new_figure(1, 2, figsize=(11, 4.4))
    window = 500
    steps = np.arange(1, 10_001)
    for length, rewards in short.items():
        ax1.plot(steps[window - 1:], ex3.moving_average(rewards.mean(axis=0), window),
                 color=LENGTH_COLORS[length], linewidth=1.5, label=f"{length} case{'s' * (length > 1)}")
    ax1.set_title(f"10 000 étapes, γ = 0.5, {n_runs} relances (glissante sur {window})",
                  fontsize=10, color=INK, loc="left")
    style(ax1, "étape", "récompense / étape")
    ax1.legend(fontsize=8.5, frameon=False)
    long_steps = np.arange(1, ex3.LONG_STEPS // ex3.LONG_BLOCK + 1) * ex3.LONG_BLOCK
    for (length, gamma), curve in curves.items():
        ax2.plot(long_steps, curve, color=LENGTH_COLORS[length], linewidth=1.5,
                 linestyle="-" if gamma == 0.1 else "--",
                 label=f"{length} case{'s' * (length > 1)}, γ = {gamma}")
    ax2.set_xscale("log")
    ax2.set_title(f"1 000 000 d'étapes, {ex3.LONG_RUNS} relances (blocs de 10 000)", fontsize=10,
                  color=INK, loc="left")
    style(ax2, "étape (échelle log)", "récompense / étape")
    ax2.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Vitesse d'apprentissage : serpent de 1 case contre 2 cases", fontsize=11, color=INK)
    save_figure(fig, "ex4_q6_comparaison.png")


# --- Q4.7 : serpents plus longs, entraînements plus longs ----------------------

Q7_GAMMA = 0.1           # parmi les meilleures sur 1e6 étapes à l'exercice 3 (ex aequo de 0.05 à 0.3)
Q7_STEPS = 1_000_000
Q7_RUNS = 16
Q7_BLOCK = 10_000
Q7_LONG_LENGTHS = [1, 2, 3, 4]
Q7_LONG_STEPS = 5_000_000
Q7_LONG_RUNS = 8
Q7_LONG_BLOCK = 50_000
Q7_WORKERS = 16          # limite la mémoire : jusqu'à ~1 million de paires par partie


def _train(task):
    """Une partie ; renvoie la récompense moyenne et le taux d'auto-intersection par bloc."""
    length, n_steps, seed, block, eps = task
    game = ShortSnake(SIZE, seed=seed, length=length)
    player = GreedyPlayer(game, eps, alpha=ALPHA, gamma=Q7_GAMMA, seed=1_000_000 + seed)
    means, hits, previous = [], [], 0
    for _ in range(n_steps // block):
        means.append(ex3.play(player, game, block).mean())
        hits.append((game.self_hits - previous) / block)
        previous = game.self_hits
    return np.array(means), np.array(hits), len(player.action_values)


def q7():
    print(f"Q4.7 : serpents de {LENGTHS} cases, γ = {Q7_GAMMA}")
    rows, curves = [], {}
    with ProcessPoolExecutor(Q7_WORKERS) as pool:
        for length in LENGTHS:
            start = time.perf_counter()
            tasks = [(length, Q7_STEPS, seed, Q7_BLOCK, EPS) for seed in range(Q7_RUNS)]
            results = list(pool.map(_train, tasks))
            means = np.array([r[0] for r in results])
            hits = np.array([r[1] for r in results])
            pairs = np.mean([r[2] for r in results])
            curves[length] = means.mean(axis=0)
            last, ci_last = mean_ci(means[:, -10:].mean(axis=1))
            hit_rate = hits[:, -10:].mean()
            random_tasks = [(length, 100_000, seed, Q7_BLOCK, 1.0) for seed in range(Q7_RUNS)]
            random_results = list(pool.map(_train, random_tasks))
            random_mean, random_ci = mean_ci([r[0].mean() for r in random_results])
            random_hits = np.mean([r[1].mean() for r in random_results])
            zero = first_step_above(curves[length], Q7_BLOCK, 0)
            n_pairs = 3 * n_long_states(SIZE, SIZE, length)
            rows.append([length, Q7_STEPS, Q7_RUNS, round(float(last), 4), round(float(ci_last), 4),
                         round(float(random_mean), 4), round(float(random_ci), 4),
                         round(float(hit_rate), 4), round(float(random_hits), 4), zero,
                         int(pairs), n_pairs, round(pairs / n_pairs, 4)])
            print(f"  {length} case(s) : {last:+.3f} ± {ci_last:.3f} sur les 1e5 dernières étapes "
                  f"(aléatoire {random_mean:+.3f}) ; auto-intersections {hit_rate:.2%} des étapes "
                  f"(aléatoire {random_hits:.2%}) ; récompense ≥ 0 à l'étape {zero} ; "
                  f"{pairs:,.0f} paires vues sur {n_pairs:,} ({pairs / n_pairs:.1%}) "
                  f"[{time.perf_counter() - start:.0f} s]")
        save_csv("ex4_q7_longueurs.csv",
                 ["longueur", "etapes", "relances", "fin", "ic95_fin", "aleatoire", "ic95_aleatoire",
                  "auto_intersections", "auto_intersections_aleatoire", "etape_recompense_0",
                  "paires_vues", "paires_possibles", "couverture"], rows)

        long_rows, long_curves = [], {}
        tasks = [(length, Q7_LONG_STEPS, seed, Q7_LONG_BLOCK, EPS)
                 for length in Q7_LONG_LENGTHS for seed in range(Q7_LONG_RUNS)]
        results = list(pool.map(_train, tasks))   # toutes les longueurs en même temps
        for i, length in enumerate(Q7_LONG_LENGTHS):
            chunk = results[i * Q7_LONG_RUNS:(i + 1) * Q7_LONG_RUNS]
            means = np.array([r[0] for r in chunk])
            long_curves[length] = means.mean(axis=0)
            last, ci_last = mean_ci(means[:, -2:].mean(axis=1))   # 100 000 dernières étapes
            zero = first_step_above(long_curves[length], Q7_LONG_BLOCK, 0)
            long_rows.append([length, Q7_LONG_STEPS, Q7_LONG_RUNS, round(float(last), 4),
                              round(float(ci_last), 4), zero])
            print(f"  {length} case(s), 5e6 étapes : {last:+.3f} ± {ci_last:.3f} sur les 1e5 dernières "
                  f"; récompense ≥ 0 à l'étape {zero}")
        save_csv("ex4_q7_5M.csv", ["longueur", "etapes", "relances", "fin", "ic95_fin",
                                   "etape_recompense_0"], long_rows)

    fig, (ax1, ax2) = new_figure(1, 2, figsize=(11, 4.4))
    steps = np.arange(1, Q7_STEPS // Q7_BLOCK + 1) * Q7_BLOCK
    for length in LENGTHS:
        ax1.plot(steps, curves[length], color=LENGTH_COLORS[length], linewidth=1.5,
                 label=f"{length} case{'s' * (length > 1)}")
    ax1.set_xscale("log")
    ax1.set_title(f"Récompense moyenne par étape ({Q7_RUNS} relances, γ = {Q7_GAMMA})", fontsize=10,
                  color=INK, loc="left")
    style(ax1, "étape (échelle log)", "récompense / étape")
    ax1.legend(fontsize=8, frameon=False, ncol=2)
    lengths = [row[0] for row in rows]
    ax2.errorbar(lengths, [row[3] for row in rows], yerr=[row[4] for row in rows], color="#2a78d6",
                 marker="o", markersize=4, linewidth=1.4, capsize=2, label="après 1e6 étapes")
    ax2.plot(lengths, [row[5] for row in rows], color=RANDOM_COLOR, linestyle="--", marker="o",
             markersize=4, linewidth=1.3, label="joueur aléatoire")
    ax2.set_xticks(lengths)
    ax2.set_title("Performance sur les 100 000 dernières étapes", fontsize=10, color=INK, loc="left")
    style(ax2, "longueur du serpent", "récompense moyenne par étape")
    ax2.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Serpents de 1 à 8 cases, 1 000 000 d'étapes d'entraînement", fontsize=11, color=INK)
    save_figure(fig, "ex4_q7_longueurs.png")

    fig, ax = new_figure(figsize=(8.5, 4.2))
    steps = np.arange(1, Q7_LONG_STEPS // Q7_LONG_BLOCK + 1) * Q7_LONG_BLOCK
    for length in Q7_LONG_LENGTHS:
        ax.plot(steps, long_curves[length], color=LENGTH_COLORS[length], linewidth=1.5,
                label=f"{length} case{'s' * (length > 1)}")
    ax.set_xscale("log")
    ax.set_title(f"5 000 000 d'étapes, {Q7_LONG_RUNS} relances, γ = {Q7_GAMMA} "
                 "(blocs de 50 000)", fontsize=10.5, color=INK, loc="left")
    style(ax, "étape (échelle log)", "récompense / étape")
    ax.legend(fontsize=8.5, frameon=False)
    save_figure(fig, "ex4_q7_5M.png")


def demo():
    """Q4.3 / Q4.7 : un serpent de 5 cases fait un carré et se mord la queue."""
    game = ShortSnake((6, 8), seed=0, length=5)
    game.restore_state(((3, 4), RIGHT, (0, 7), ((3, 3), (3, 2), (3, 1), (3, 0))))
    print(game, end="\n\n")
    for direction in (DOWN, LEFT, UP):
        reward = game.move(direction)
        print(f"move({direction}) -> {reward:+d}")
        print(game, end="\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["demo", "q2", "q5", "q6", "q7", "all"])
    args = parser.parse_args()
    if args.command == "demo":
        demo()
        return
    start = time.perf_counter()
    if args.command in ("q2", "q5", "q6", "all"):
        with ProcessPoolExecutor() as pool:
            if args.command in ("q2", "all"):
                q2(pool)
            if args.command in ("q5", "all"):
                q5(pool)
            if args.command in ("q6", "all"):
                q6(pool)
    if args.command in ("q7", "all"):
        q7()
    print(f"Terminé en {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
