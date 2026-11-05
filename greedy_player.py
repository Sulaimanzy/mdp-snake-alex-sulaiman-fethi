"""Exercice 3 : joueur Q-Learning utilisant la stratégie epsilon-greedy.

Repris de la classe GreedyPlayer du TP Bandit manchot. La structure est
conservée (get_action, _greedy_action, _random_action, reward) ; le
constructeur, _greedy_action et reward tiennent maintenant compte de l'état.
"""

import random

from mdp import actions


class GreedyPlayer:
    """Apprend Q(état, action) par Q-Learning en jouant à ShortSnake."""

    def __init__(self, game, eps, alpha=0.5, gamma=0.5, seed=None):
        if not 0 <= eps <= 1:
            raise ValueError("Epsilon doit être compris entre 0 et 1.")
        if not 0 < alpha <= 1:
            raise ValueError("Alpha doit être compris dans ]0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma doit être compris entre 0 et 1.")

        self.game = game          # Q3.5 -- instance de ShortSnake
        self.eps = eps
        self.alpha = alpha        # taux d'apprentissage
        self.gamma = gamma        # facteur d'actualisation
        self.action_values = {}   # Q3.6 -- Q[(état, action)], 0 si jamais mis à jour
        self.eval_count = {}      # nombre de mises à jour de chaque paire (état, action)
        self.rng = random.Random(seed)
        self.state = None         # état dans lequel la dernière action a été choisie

    def get_action(self):
        """Explore avec la probabilité eps, sinon exploite les estimations."""
        self.state = self.game.save_state()
        explore = self.rng.random() < self.eps
        if explore:
            return self._random_action(self.state)
        return self._greedy_action(self.state)

    def _value(self, state, action):
        return self.action_values.get((state, action), 0.0)

    # Q3.7 -- action gloutonne dans l'état courant
    def _greedy_action(self, state):
        """Choisit une action de valeur Q(state, a) maximale, au hasard en cas d'égalité."""
        valeurs = [(action, self._value(state, action)) for action in actions(state)]
        meilleure_valeur = max(valeur for _, valeur in valeurs)
        meilleures_actions = [
            action for action, valeur in valeurs if valeur == meilleure_valeur
        ]
        return self.rng.choice(meilleures_actions)

    def _random_action(self, state):
        """Choisit uniformément parmi les actions possibles dans `state`."""
        return self.rng.choice(actions(state))

    # Q3.8 -- mise à jour du Q-Learning
    def reward(self, action, value):
        """Met à jour Q(s, a) après game.move(action), qui a rapporté `value`.

        Q(s, a) <- (1 - alpha) Q(s, a) + alpha (r + gamma max_a' Q(s', a'))
        """
        if self.state is None:
            raise RuntimeError("get_action doit être appelée avant reward.")
        if action not in actions(self.state):
            raise ValueError(f"action impossible dans l'état {self.state} : {action!r}")

        etat_suivant = self.game.save_state()
        meilleure_suite = max(self._value(etat_suivant, a) for a in actions(etat_suivant))
        cible = value + self.gamma * meilleure_suite

        cle = (self.state, action)
        self.action_values[cle] = (
            (1 - self.alpha) * self.action_values.get(cle, 0.0) + self.alpha * cible
        )
        self.eval_count[cle] = self.eval_count.get(cle, 0) + 1
        self.state = None   # une seule mise à jour par action choisie
