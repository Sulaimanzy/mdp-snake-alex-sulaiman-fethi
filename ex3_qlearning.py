"""Exercice 3 : programme principal du Q-Learning et expériences.

    py ex3_qlearning.py play [--steps 500000] [--show 15]  # Q3.9 : l'IA apprend puis joue
    py ex3_qlearning.py q10                                # une partie de 1000 étapes
    py ex3_qlearning.py q11                                # nombre de relances
    py ex3_qlearning.py q12                                # gamma dans {0, 0.1, 0.5, 0.9, 1}
    py ex3_qlearning.py q13                                # recherche du meilleur gamma
    py ex3_qlearning.py q14                                # même recherche sur 10000 étapes
    py ex3_qlearning.py long                               # pour aller plus loin : 1e6 étapes
    py ex3_qlearning.py all                                # toutes les expériences

Les figures vont dans rapport/figures/, les résultats chiffrés dans resultats/.
"""

import argparse
import csv
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from greedy_player import GreedyPlayer
from mdp import actions
from snake import ShortSnake

SIZE = 10           # grille 10 x 10
EPS = 0.1           # probabilité d'exploration
ALPHA = 0.5         # taux d'apprentissage
GAMMAS = [0, 0.1, 0.5, 0.9, 1]
SWEEP = [round(0.05 * i, 2) for i in range(21)]   # 0, 0.05, ..., 1
N_RUNS = 1000       # relances pour lisser le bruit (choisi à la question 11)
N_RUNS_LONG = 500   # relances pour les parties de 10000 étapes

ROOT = pathlib.Path(__file__).resolve().parent
FIGURES = ROOT / "rapport" / "figures"
RESULTS = ROOT / "resultats"


# --- Q3.9 : programme principal ---------------------------------------------

def play(player, game, n_steps):
    """Fait jouer l'IA pendant n_steps étapes ; renvoie la récompense de chaque étape."""
    rewards = np.empty(n_steps, dtype=np.int8)
    for t in range(n_steps):
        action = player.get_action()
        value = game.move(action)
        player.reward(action, value)
        rewards[t] = value
    return rewards


def run(gamma, n_steps, seed, eps=EPS, alpha=ALPHA):
    """Une partie complète avec une IA neuve (graines fixées pour la reproductibilité)."""
    game = ShortSnake(SIZE, seed=seed)
    player = GreedyPlayer(game, eps, alpha=alpha, gamma=gamma, seed=1_000_000 + seed)
    return play(player, game, n_steps)


def _run_chunk(task):
    gamma, n_steps, seeds, eps, alpha = task
    return np.stack([run(gamma, n_steps, int(seed), eps, alpha) for seed in seeds])


def run_many(pool, gamma, n_steps, n_runs, eps=EPS, alpha=ALPHA, first_seed=0):
    """Relance l'IA n_runs fois (en parallèle) ; renvoie un tableau (n_runs, n_steps)."""
    if n_runs < 1 or n_steps < 1:
        raise ValueError("il faut au moins une relance et une étape")
    seeds = np.arange(first_seed, first_seed + n_runs)
    n_chunks = min(n_runs, 4 * (os.cpu_count() or 1))
    tasks = [(gamma, n_steps, chunk, eps, alpha) for chunk in np.array_split(seeds, n_chunks)]
    return np.concatenate(list(pool.map(_run_chunk, tasks)))


# --- Statistiques -------------------------------------------------------------

_T95_SMALL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def t95(df):
    """Quantile 97,5 % de la loi de Student : table exacte jusqu'à 6 degrés de liberté,
    développement de Cornish-Fisher au-delà (erreur < 0,5 %)."""
    if df in _T95_SMALL:
        return _T95_SMALL[df]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def mean_ci(values):
    """Moyenne et demi-largeur de l'intervalle de confiance à 95 % (loi de Student)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.mean(), float("nan")
    return values.mean(), t95(len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))


def total_stats(rewards):
    """Récompense totale moyenne et demi-largeur de son intervalle de confiance à 95 %."""
    return mean_ci(rewards.sum(axis=1, dtype=np.int64))


def paired_difference(rewards, reference):
    """Écart moyen de récompense totale avec `reference`, partie par partie (mêmes graines)."""
    return mean_ci(rewards.sum(axis=1, dtype=np.int64) - reference.sum(axis=1, dtype=np.int64))


def moving_average(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_csv(name, header, rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def new_figure(*args, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def save_figure(fig, name):
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  figure :", FIGURES / name)


# Couleurs : une teinte fixe par valeur de gamma, gris pour le joueur aléatoire
COLORS = {0: "#2a78d6", 0.1: "#eb6834", 0.5: "#1baf7a", 0.9: "#eda100", 1: "#e87ba4"}
RANDOM_COLOR = "#898781"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def style(ax, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def gamma_label(gamma):
    return f"γ = {gamma:g}"


def identical_runs(rewards, reference):
    """Nombre de parties strictement identiques (même graine) à celles de `reference`."""
    return int(np.all(rewards == reference, axis=1).sum())


# --- Q3.10 : une partie de 1000 étapes --------------------------------------

def q10(pool):
    print("Q3.10 : une partie de 1000 étapes, γ = 0.5")
    rewards = run(0.5, 1000, seed=0)
    steps = np.arange(1, len(rewards) + 1)
    print(f"  score final {rewards.sum()} ; {int((rewards == 10).sum())} étapes à +10, "
          f"{int((rewards == -10).sum())} étapes à -10")

    fig, (ax1, ax2) = new_figure(2, 1, figsize=(8.5, 5.6), sharex=True)
    window = 50
    ax1.plot(steps[window - 1:], moving_average(rewards, window), color=COLORS[0.5], linewidth=1.6)
    ax1.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax1.set_title("Récompense moyenne sur les 50 dernières étapes", fontsize=10, color=INK, loc="left")
    style(ax1, ylabel="récompense / étape")
    ax2.plot(steps, np.cumsum(rewards), color=COLORS[0.5], linewidth=1.6)
    ax2.set_title("Score cumulé (somme des récompenses)", fontsize=10, color=INK, loc="left")
    style(ax2, xlabel="étape", ylabel="score")
    fig.suptitle("Une partie de 1000 étapes (γ = 0.5, ε = 0.1, α = 0.5)", fontsize=11, color=INK)
    save_figure(fig, "ex3_q10_une_partie.png")


# --- Q3.11 : combien de relances ? ------------------------------------------

def q11(pool):
    print("Q3.11 : nombre de relances pour lisser le bruit (γ = 0.5, 1000 étapes)")
    batch_a = run_many(pool, 0.5, 1000, 1000)
    batch_b = run_many(pool, 0.5, 1000, 1000, first_seed=100_000)
    window = 20
    rows = []
    fig, axes = new_figure(2, 2, figsize=(9, 5.8), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, (1, 10, 100, 1000)):
        curve_a = moving_average(batch_a[:k].mean(axis=0), window)
        curve_b = moving_average(batch_b[:k].mean(axis=0), window)
        rms = float(np.sqrt(np.mean((curve_a - curve_b) ** 2)))
        mean_total, ci = total_stats(batch_a[:k])
        rows.append([k, round(rms, 4), round(float(mean_total), 1), round(float(ci), 1)])
        print(f"  {k:5} relances : écart RMS entre deux séries = {rms:.3f} ; "
              f"total moyen = {mean_total:.1f} ± {ci:.1f}")
        steps = np.arange(window, 1001)
        ax.plot(steps, curve_a, color=COLORS[0], linewidth=1.3, label="série A")
        ax.plot(steps, curve_b, color=COLORS[0.1], linewidth=1.3, label="série B")
        ax.set_title(f"{k} relance{'s' if k > 1 else ''} - écart RMS {rms:.2f}", fontsize=9.5,
                     color=INK, loc="left")
        style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("étape", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("récompense / étape", color=MUTED, fontsize=9)
    fig.suptitle("Récompense moyenne de deux séries indépendantes de relances "
                 "(moyenne glissante sur 20 étapes)", fontsize=10.5, color=INK)
    save_figure(fig, "ex3_q11_relances.png")
    save_csv("ex3_q11_relances.csv", ["relances", "ecart_rms", "total_moyen", "ic95"], rows)


# --- Q3.12 : gamma dans {0, 0.1, 0.5, 0.9, 1} --------------------------------

def compare_gammas(pool, n_steps, n_runs, window, name, title):
    results = {gamma: run_many(pool, gamma, n_steps, n_runs) for gamma in GAMMAS}
    baseline = run_many(pool, 0.5, n_steps, n_runs, eps=1.0)
    rows = []
    for gamma, rewards in results.items():
        mean_total, ci = total_stats(rewards)
        same = identical_runs(rewards, results[0.5])
        rows.append([gamma, n_steps, n_runs, round(float(mean_total), 1), round(float(ci), 1), same])
        print(f"  {gamma_label(gamma):9} total moyen {mean_total:9.1f} ± {ci:5.1f}   "
              f"parties identiques à γ = 0.5 : {same}/{n_runs}")
    mean_total, ci = total_stats(baseline)
    rows.append(["aleatoire", n_steps, n_runs, round(float(mean_total), 1), round(float(ci), 1), ""])
    print(f"  aléatoire total moyen {mean_total:9.1f} ± {ci:5.1f}")

    fig, (ax1, ax2) = new_figure(1, 2, figsize=(11, 4.4))
    steps = np.arange(1, n_steps + 1)
    for gamma, rewards in results.items():
        ax1.plot(steps[window - 1:], moving_average(rewards.mean(axis=0), window),
                 color=COLORS[gamma], linewidth=1.5, label=gamma_label(gamma))
        ax2.plot(steps, np.cumsum(rewards.mean(axis=0)), color=COLORS[gamma], linewidth=1.5,
                 label=gamma_label(gamma))
    for ax, curve in ((ax1, moving_average(baseline.mean(axis=0), window)),
                      (ax2, np.cumsum(baseline.mean(axis=0)))):
        x = steps[window - 1:] if ax is ax1 else steps
        ax.plot(x, curve, color=RANDOM_COLOR, linewidth=1.3, linestyle="--", label="aléatoire (ε = 1)")
    ax1.set_title(f"Récompense moyenne par étape (glissante sur {window})", fontsize=10,
                  color=INK, loc="left")
    ax2.set_title("Récompense cumulée moyenne", fontsize=10, color=INK, loc="left")
    style(ax1, "étape", "récompense / étape")
    style(ax2, "étape", "récompense cumulée")
    ax2.legend(fontsize=8.5, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK)
    save_figure(fig, name)
    return results, rows


def _visit_stats(n_steps, gamma=0.5, seed=0):
    """Paires mises à jour, et part des mises à jour où max_a Q(s', a) ≠ 0 (seul cas où γ compte)."""
    game = ShortSnake(SIZE, seed=seed)
    player = GreedyPlayer(game, EPS, alpha=ALPHA, gamma=gamma, seed=1_000_000 + seed)
    gamma_matters = 0
    for _ in range(n_steps):
        action = player.get_action()
        value = game.move(action)
        next_state = game.save_state()
        if max(player._value(next_state, a) for a in actions(next_state)) != 0:
            gamma_matters += 1
        player.reward(action, value)
    counts = np.array(list(player.eval_count.values()))
    return [n_steps, len(counts), round(float((counts > 1).mean()), 4), int(counts.max()),
            round(gamma_matters / n_steps, 4)]


def q12(pool):
    print(f"Q3.12 : γ dans {GAMMAS}, 1000 étapes, {N_RUNS} relances")
    _, rows = compare_gammas(pool, 1000, N_RUNS, 50, "ex3_q12_gammas.png",
                             f"Q-Learning, 1000 étapes, moyenne de {N_RUNS} relances")
    save_csv("ex3_q12_gammas.csv",
             ["gamma", "etapes", "relances", "total_moyen", "ic95", "identiques_a_0.5"], rows)

    print("  paires (état, action) mises à jour au cours d'une partie (graine 0, γ = 0.5) :")
    visits = list(pool.map(_visit_stats, (1000, 10_000, 100_000)))
    for n_steps, n_pairs, multi, top, matters in visits:
        print(f"    {n_steps:6} étapes : {n_pairs:6} paires, {multi:.1%} mises à jour plus "
              f"d'une fois, au plus {top} fois ; max Q(s', ·) ≠ 0 dans {matters:.2%} des mises à jour")
    save_csv("ex3_visites.csv", ["etapes", "paires", "part_plus_d_une_fois", "maximum",
                                 "part_max_q_suivant_non_nul"], visits)


# --- Q3.13 / Q3.14 : recherche du meilleur gamma ----------------------------

SWEEP_HEADER = ["gamma", "etapes", "relances", "total_moyen", "ic95", "identiques_a_0.5",
                "ecart_apparie_a_0.5", "ic95_ecart", "verdict"]


def verdict(diff, ci):
    """Un écart est significatif si son intervalle de confiance à 95 % ne contient pas 0."""
    if diff == 0:
        return "aucun écart"
    if not np.isfinite(ci):
        return "indéterminé"
    return "significatif" if abs(diff) > ci else "non significatif"


def sweep(pool, n_steps, n_runs, gammas=SWEEP):
    """Balaye gamma. Comme toutes les valeurs partagent les mêmes graines, chaque valeur est
    comparée à γ = 0.5 partie par partie (écart apparié), ce qui retire le bruit commun."""
    reference = run_many(pool, 0.5, n_steps, n_runs)
    rows, means = [], {}
    for gamma in gammas:
        rewards = reference if gamma == 0.5 else run_many(pool, gamma, n_steps, n_runs)
        mean_total, ci = total_stats(rewards)
        diff, ci_diff = paired_difference(rewards, reference)
        means[gamma] = mean_total
        rows.append([gamma, n_steps, n_runs, round(float(mean_total), 1), round(float(ci), 1),
                     identical_runs(rewards, reference), round(float(diff), 2),
                     round(float(ci_diff), 2), verdict(diff, ci_diff)])
    best_mean = max(means.values())
    best = [gamma for gamma, mean in means.items() if mean == best_mean]
    for row in rows:
        mark = "  <- meilleur" if row[0] in best else ""
        print(f"  γ = {row[0]:<4} total moyen {row[3]:9.1f} ± {row[4]:5.1f}   "
              f"identiques à γ = 0.5 : {row[5]}/{n_runs}   écart apparié {row[6]:+.2f} ± {row[7]:.2f} "
              f"({row[8]}){mark}")
    print(f"  meilleure moyenne : γ = {best}" + (" (ex aequo)" if len(best) > 1 else ""))
    return rows, best


def plot_sweep(rows_by_horizon, name, title):
    fig, ax = new_figure(figsize=(8.5, 4.2))
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (label, rows), color in zip(rows_by_horizon, palette):
        gammas = [row[0] for row in rows]
        per_step = [row[3] / row[1] for row in rows]
        ci = [row[4] / row[1] for row in rows]
        ax.errorbar(gammas, per_step, yerr=ci, color=color, marker="o", markersize=4,
                    linewidth=1.4, capsize=2, label=label)
    style(ax, "facteur d'actualisation γ", "récompense moyenne par étape")
    if len(rows_by_horizon) > 1:   # une seule série : le titre suffit
        ax.legend(fontsize=8.5, frameon=False, loc="center right")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    save_figure(fig, name)


def q13(pool):
    print(f"Q3.13 : recherche de γ sur [0, 1] (pas 0.05), 1000 étapes, {N_RUNS} relances")
    rows, best = sweep(pool, 1000, N_RUNS)
    save_csv("ex3_q13_recherche.csv", SWEEP_HEADER, rows)
    plot_sweep([("1000 étapes", rows)], "ex3_q13_recherche_gamma.png",
               "Récompense moyenne par étape selon γ, 1000 étapes (barres : IC à 95 %)")
    return rows


def q14(pool, rows_1000=None):
    print(f"Q3.14 : même recherche sur 10000 étapes, {N_RUNS_LONG} relances")
    rows, best = sweep(pool, 10_000, N_RUNS_LONG)
    save_csv("ex3_q14_recherche_10000.csv", SWEEP_HEADER, rows)
    if rows_1000 is None:
        rows_1000, _ = sweep(pool, 1000, N_RUNS)
    plot_sweep([("1000 étapes", rows_1000), ("10000 étapes", rows)], "ex3_q14_recherche_10000.png",
               "Récompense moyenne par étape selon γ (barres : IC à 95 %)")
    compare_gammas(pool, 10_000, N_RUNS_LONG, 500, "ex3_q14_gammas_10000.png",
                   f"Q-Learning, 10000 étapes, moyenne de {N_RUNS_LONG} relances")


# --- Pour aller plus loin : entraînements beaucoup plus longs --------------

LONG_STEPS = 1_000_000
LONG_BLOCK = 10_000
LONG_RUNS = 32
LONG_GAMMAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]


def _run_blocks(task):
    gamma, n_steps, seed, block = task
    return run(gamma, n_steps, seed).reshape(-1, block).mean(axis=1)


MAX_Q_CHECKPOINTS = (10_000, 100_000, 300_000, 1_000_000)


def _max_q(gamma, seed=0):
    """max |Q| au fil d'une même partie (graine 0), à chaque point de MAX_Q_CHECKPOINTS."""
    game = ShortSnake(SIZE, seed=seed)
    player = GreedyPlayer(game, EPS, alpha=ALPHA, gamma=gamma, seed=1_000_000 + seed)
    done, values = 0, []
    for checkpoint in MAX_Q_CHECKPOINTS:
        play(player, game, checkpoint - done)
        done = checkpoint
        values.append(round(max(abs(v) for v in player.action_values.values()), 2))
    return [gamma, *values]


def long_horizon(pool):
    print(f"Pour aller plus loin : {LONG_STEPS} étapes, {LONG_RUNS} relances")
    curves, rows = {}, []
    for gamma in LONG_GAMMAS:
        tasks = [(gamma, LONG_STEPS, seed, LONG_BLOCK) for seed in range(LONG_RUNS)]
        blocks = np.array(list(pool.map(_run_blocks, tasks)))   # (relances, blocs)
        curves[gamma] = blocks.mean(axis=0)
        mean_last, ci = mean_ci(blocks[:, -10:].mean(axis=1))      # 100 000 dernières étapes
        mean_total, ci_total = mean_ci(blocks.mean(axis=1))
        rows.append([gamma, round(float(mean_total), 4), round(float(ci_total), 4),
                     round(float(mean_last), 4), round(float(ci), 4)])
        print(f"  γ = {gamma:<4} moyenne sur 1e6 étapes {mean_total:6.3f} ± {ci_total:.3f}   "
              f"sur les 1e5 dernières {mean_last:6.3f} ± {ci:.3f}   pic {curves[gamma].max():.3f}")
    save_csv("ex3_long_horizon.csv",
             ["gamma", "moyenne_1e6", "ic95", "moyenne_1e5_dernieres", "ic95_dernieres"], rows)
    max_q = list(pool.map(_max_q, (0.5, 0.9, 1)))
    print("  max |Q| après " + ", ".join(f"{c:,}" for c in MAX_Q_CHECKPOINTS) + " étapes (graine 0) :")
    for gamma, *values in max_q:
        bound = f"borne 10/(1-γ) = {10 / (1 - gamma):.0f}" if gamma < 1 else "pas de borne"
        print(f"    γ = {gamma:<4} {values}   ({bound})")
    save_csv("ex3_max_q.csv", ["gamma", *[f"max_abs_q_{c}" for c in MAX_Q_CHECKPOINTS]], max_q)

    fig, (ax1, ax2) = new_figure(1, 2, figsize=(11, 4.4))
    steps = np.arange(1, LONG_STEPS // LONG_BLOCK + 1) * LONG_BLOCK
    for gamma in GAMMAS:
        ax1.plot(steps, curves[gamma], color=COLORS[gamma], linewidth=1.5, label=gamma_label(gamma))
    ax1.set_xscale("log")
    ax1.set_title("Récompense moyenne par étape (blocs de 10 000)", fontsize=10, color=INK, loc="left")
    style(ax1, "étape (échelle log)", "récompense / étape")
    ax1.legend(fontsize=8.5, frameon=False)
    gammas = [row[0] for row in rows]
    ax2.errorbar(gammas, [row[3] for row in rows], yerr=[row[4] for row in rows],
                 color="#2a78d6", marker="o", markersize=4, linewidth=1.4, capsize=2)
    ax2.set_title("Performance sur les 100 000 dernières étapes", fontsize=10, color=INK, loc="left")
    style(ax2, "facteur d'actualisation γ", "récompense moyenne par étape")
    fig.suptitle(f"Q-Learning sur 1 000 000 d'étapes, moyenne de {LONG_RUNS} relances",
                 fontsize=11, color=INK)
    save_figure(fig, "ex3_long_horizon.png")


# --- Q3.9 : regarder l'IA jouer ----------------------------------------------

def watch(n_steps, n_show, gamma, seed):
    if n_steps < 1:
        raise SystemExit("--steps doit être au moins 1")
    game = ShortSnake(SIZE, seed=seed)
    player = GreedyPlayer(game, EPS, alpha=ALPHA, gamma=gamma, seed=1_000_000 + seed)
    start = time.perf_counter()
    rewards = play(player, game, n_steps)
    last = rewards[-1000:]
    print(f"Entraînement : {n_steps} étapes en {time.perf_counter() - start:.1f} s, "
          f"score {game.score}, récompense moyenne sur les {len(last)} dernières étapes "
          f"{last.mean():+.3f}, {len(player.action_values)} paires (état, action) apprises")
    player.eps = 0  # on regarde la politique apprise, sans exploration
    for step in range(1, n_show + 1):
        action = player.get_action()
        value = game.move(action)
        player.reward(action, value)
        print(f"\nÉtape {step} : {action} -> {value:+d}")
        print(game)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["play", "q10", "q11", "q12", "q13", "q14", "long", "all"])
    parser.add_argument("--steps", type=int, default=500_000, help="étapes d'entraînement (play)")
    parser.add_argument("--show", type=int, default=15, help="étapes affichées (play)")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "play":
        watch(args.steps, args.show, args.gamma, args.seed)
        return

    start = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        if args.command in ("q10", "all"):
            q10(pool)
        if args.command in ("q11", "all"):
            q11(pool)
        if args.command in ("q12", "all"):
            q12(pool)
        rows_1000 = q13(pool) if args.command in ("q13", "all") else None
        if args.command in ("q14", "all"):
            q14(pool, rows_1000)
        if args.command in ("long", "all"):
            long_horizon(pool)
    print(f"Terminé en {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
