# TP MDP : Snake

ESGI - AL5 / T1 - Apprentissage par renforcement (Sylvain Gault).

Groupe : RUSESCU Daniel Alexandru, BAH Sulaiman, SEDJAI Fethi.

Variante de Snake où le serpent ne grandit pas et ne meurt pas : il gagne 10 points
en mangeant un fruit et en perd 10 en traversant un bord. Le jeu est modélisé comme un
processus de décision markovien puis résolu par Q-Learning.

Le rapport est dans [`rapport/RUSESCU_BAH_SEDJAI_rapport_TP_Snake.pdf`](rapport/RUSESCU_BAH_SEDJAI_rapport_TP_Snake.pdf).

## Installation

Python 3.9 au minimum (exigé par matplotlib 3.8) ; développé et testé avec Python 3.14
sous Windows. Toutes les commandes se lancent depuis la racine du dépôt, dans un
environnement virtuel.

Windows (PowerShell) :

```powershell
py -m venv .venv                                            # ou : python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # autorise l'activation pour cette fenêtre
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dans l'invite de commandes (cmd), l'activation se fait avec `.venv\Scripts\activate.bat`.

macOS / Linux (sous Debian ou Ubuntu, installer d'abord le paquet `python3-venv`) :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Lancer le projet

Une fois l'environnement activé, sur tous les systèmes :

```bash
python -m unittest discover tests      # tests
python ex1_demo.py                     # démonstration du jeu
python ex1_demo.py --play              # jouer au clavier (z q s d + Entrée)
python ex2_mdp.py                      # exercice 2 : réponses et figure du MDP
python ex3_qlearning.py play           # exercice 3 : l'IA apprend puis joue
python ex3_qlearning.py all            # exercice 3 : toutes les expériences (≈ 4 min sur 32 cœurs)
python ex4_long_snake.py demo          # exercice 4 : un serpent de 5 cases se mord la queue
python ex4_long_snake.py all           # exercice 4 : toutes les expériences (≈ 9 min sur 32 cœurs, ≈ 7 Go de RAM)
python rapport/build_pdf.py            # régénérer le rapport PDF
```

Les commandes `all` réécrivent les figures (`rapport/figures/`) et les résultats
(`resultats/`). Les graines aléatoires sont fixées : les résultats sont identiques d'une
machine à l'autre, quel que soit le nombre de cœurs.

La génération du PDF imprime la page avec Edge, Chrome ou Chromium, cherchés aux
emplacements habituels sous Windows, macOS et Linux. Pour en indiquer un autre :

```text
PowerShell : $env:PDF_BROWSER = "C:\chemin\vers\chrome.exe"
cmd        : set PDF_BROWSER=C:\chemin\vers\chrome.exe
bash / zsh : export PDF_BROWSER="/chemin/vers/chrome"
```
