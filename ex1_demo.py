"""Exercice 1 : démonstration du jeu de base.

    py ex1_demo.py           # partie scriptée (bords, demi-tour, fruit)
    py ex1_demo.py --play    # jouer au clavier (z q s d + Entrée)
"""

import sys

from snake import DOWN, LEFT, RIGHT, UP, ShortSnake

KEYS = {"z": UP, "s": DOWN, "q": LEFT, "d": RIGHT, "": None}


def scripted_demo():
    game = ShortSnake((5, 8), seed=42)
    print("État initial :")
    print(game, end="\n\n")

    steps = [
        (None, "tout droit"),
        (LEFT, "demi-tour interdit : on continue à droite"),
        (UP, "on tourne"),
        (RIGHT, "on tourne"),
        (UP, "on mange le fruit : +10"),
        (UP, "bord du haut franchi : on réapparaît en bas, -10"),
        (None, "tout droit"),
    ]
    for step, (direction, comment) in enumerate(steps, start=1):
        reward = game.move(direction)
        print(f"Étape {step} : move({direction}) -> récompense {reward:+d} ({comment})")
        print(game, end="\n\n")


def play():
    game = ShortSnake(10)
    print("z/q/s/d pour tourner, Entrée pour continuer tout droit, x pour quitter.")
    while True:
        print(game)
        key = input("> ").strip().lower()
        if key == "x":
            break
        if key not in KEYS:
            print("Touche inconnue.")
            continue
        reward = game.move(KEYS[key])
        if reward:
            print(f"Récompense : {reward:+d}")


if __name__ == "__main__":
    if "--play" in sys.argv:
        play()
    else:
        scripted_demo()
