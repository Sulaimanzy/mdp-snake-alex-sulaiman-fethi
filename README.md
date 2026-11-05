# TP MDP : Snake

ESGI - AL5 / T1 — Apprentissage par renforcement (Sylvain Gault).

Groupe : RUSESCU Daniel Alexandru, BAH Sulaiman, SEDJAI Fethi.

Variante de Snake où le serpent ne grandit pas et ne meurt pas : il gagne 10 points
en mangeant un fruit et en perd 10 en traversant un bord. Le jeu est modélisé comme un
processus de décision markovien puis résolu par Q-Learning.

Le rapport est dans [`rapport/RUSESCU_BAH_SEDJAI_rapport_TP_Snake.pdf`](rapport/RUSESCU_BAH_SEDJAI_rapport_TP_Snake.pdf).

## Lancer le projet

```bash
py -m pip install -r requirements.txt   # dépendances (génération du rapport)
py -m unittest discover tests           # tests
py ex1_demo.py                          # démonstration du jeu
py ex1_demo.py --play                   # jouer au clavier (z q s d + Entrée)
py rapport/build_pdf.py                 # régénérer le rapport PDF
```
