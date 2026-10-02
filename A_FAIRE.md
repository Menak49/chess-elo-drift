# Ce qu'il reste à faire

État au 30 septembre 2026, 16h40. L'étude a été refaite pour couvrir 2014 à 2026, sur chess.com et Lichess, avec une conversion 2026 entre les quatre pools de classement. Le code est terminé et testé : 281 tests passent. La collecte Lichess est quasiment finie (le processus tourne encore pour ramasser les dernières parties rares). Il reste surtout l'analyse moteur — bien plus qu'estimé la dernière fois, parce que la recollecte a fait grossir le corpus — et le rapport final.

## Où on en est

| Étape | État | Détail |
|---|---|---|
| Collecte chess.com 2014–2026 | ✅ terminée | 10 414 parties. Toutes les cases sont pleines, sauf le rapide 600–999 et 2000+ avant 2020 : ces classements étaient réellement rares quand « rapide » voulait dire 15 min et plus. |
| Collecte Lichess 2014–2026 | ✅ quasi terminée | 6 957 parties. Toutes les cases sont à leur cible (48/case) sauf trois, attendues : 600 blitz en 2014 (22) et 2016 (23), réellement rares ; 600 vide en 2018–2019, normal puisque le plancher Lichess était à 800 avant fin 2019. Un processus de collecte (lancé lors d'une session précédente, `collect --platform lichess --years 2014-2024,2026`) tournait encore à 16h40 pour ramasser les dernières parties atteignables sur 2016–2017 ; il s'arrêtera de lui-même. **Ne pas relancer de collecte Lichess tant qu'il tourne** (une seule requête à la fois, voir plus bas). |
| Analyse moteur | ⚠️ partielle | 27 951 lignes (deux par partie, une par joueur), soit environ 12 285 parties uniques analysées. **Il reste environ 3 400 parties** à passer au moteur : chess.com 2017–2026 (surtout 2018–2026, ~250 à 345 parties par an) et Lichess 2014–2016, 2018, 2024 et 2025 (~200 parties par an, sauf 2025 : seulement 12 manquantes). Le chiffre est plus élevé que l'estimation précédente (~2 200) car la recollecte a ajouté des parties chess.com après le dernier passage du moteur. |
| Sondage de conversion | ✅ terminé | 8 486 comptes chess.com, 23 518 comptes Lichess et 611 comptes liés entre les deux sites (`data/raw/conversion/`). |
| Rapport final | ❌ pas généré | `reports/findings.md` contient encore l'**ancienne** étude (2018–19 contre 2024–25). |
| Commit | ❌ rien de commité | Voir la dernière section. |

## Les étapes restantes, dans l'ordre

Toutes les commandes se lancent depuis la racine du dépôt. Chaque étape peut être interrompue puis relancée : elle reprend là où elle s'est arrêtée.

### 1. Laisser la collecte Lichess finir toute seule (quasi fait)

Rien à lancer : un processus est déjà en cours depuis une session précédente et couvre tout ce qu'il fallait. Vérifier juste qu'il n'y en a pas deux à la fois avant de faire quoi que ce soit d'autre :

```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine"
```

- Lichess n'accepte **qu'une requête à la fois**. Ne jamais lancer une deuxième collecte Lichess pendant qu'une tourne déjà.
- Évitez aussi de tuer la collecte au milieu d'une requête : un flux abandonné peut bloquer l'adresse IP environ 1 h. Si ça arrive, les requêtes renvoient 429 « only run 1 request at a time ». Il suffit d'attendre une heure.
- Une année dont les cases restantes sont impossibles à remplir s'arrête d'elle-même après 300 requêtes sans progrès. C'est voulu.
- S'il n'y a vraiment plus de processus et qu'il manque encore des cases (hors les trois exceptions attendues listées plus haut), relancer avec les années concernées, par exemple `collect --platform lichess --years 2016-2017`.

### 2. Finir l'analyse moteur — environ 3 400 parties restantes (30–60 min avec 4 workers)

```bash
PYTHONPATH=src python -m chess_elo_drift.cli evaluate --workers 4
```

- **Mémoire** : Claude Code a arrêté des tâches plusieurs fois parce que la machine manquait de RAM (autour de 2,5 Go disponibles sur 15,8 Go au 30/09 16h, surtout à cause de Brave).
  - Avec 4 workers, ça passe. Avec 8 ou 14, c'est plus rapide si beaucoup de RAM est libre.
  - Fermez des onglets avant de lancer.
  - Vérifier la RAM libre avant de lancer : `powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory,TotalVisibleMemorySize"`.
- Attendre que la collecte Lichess (étape 1) soit bien terminée avant de lancer l'analyse moteur, pour ne pas cumuler la charge des deux en même temps sur une machine déjà limitée en RAM.
- Si un worker se bloque et qu'aucune partie ne se termine pendant 10 minutes, la commande s'arrête d'elle-même. Il suffit de la relancer.
- 5 parties chess.com resteront toujours non analysées, et c'est normal. Ce sont des tournois thématiques qui partent d'une position imposée : elles sont exclues.

### 3. Générer le rapport (environ 1 min)

```bash
PYTHONPATH=src python -m chess_elo_drift.cli report
```

Cette commande écrit `reports/findings.md`, `reports/tables/*.csv` et `reports/figures/*.png`. Tous les chiffres sont calculés depuis les données : rien n'est tapé à la main.

### 4. Relire le rapport avant de le publier

Points à vérifier en particulier :

- **Conversion chess.com rapide ↔ blitz, en haut de l'échelle.** C'est la conversion la moins sûre.
  - Au 1er essai, le rapide 2000 donnait environ blitz 1800 d'après le sondage, contre 1883 d'après les instantanés mensuels (33 personnes seulement).
  - Vérifier la section « Second source » et l'écart avec ChessGoals (+75 à +120 points au-dessus de 1500).
- **Comparaison chess.com / Lichess par année.** Elle n'a pas encore pu tourner sur les vraies données, faute de parties Lichess analysées au moment de l'essai. Vérifier que les écarts sont stables d'une année à l'autre, et qu'ils sont cohérents avec la conversion par comptes liés (tableau « Accuracy cross-check »).
- **Rupture de 2020 sur chess.com rapide.** Le 10|0 est passé en rapide le 10/09/2020. Regarder les tendances « avant / après » plutôt que la tendance unique.
- **Les figures** dans `reports/figures/` : lisibilité, et cases vides signalées comme telles.

### 5. Commiter

Rien n'a encore été commité. Il y a :

- les fichiers modifiés (`git status`) ;
- les nouveaux modules, non suivis : `http.py`, `lichess/`, `conversion/`, `collection/sources.py`, `analysis/{yearly,sites,surveys,conversion}.py`, `tests/test_lichess.py` et `tests/test_survey.py`.

`data/` et `logs/` sont ignorés par git. `reports/` est suivi, comme prévu par votre `.gitignore`. Relancer `python -m pytest` avant de commiter.

## À savoir pour reprendre

- **Lancer avec `PYTHONPATH=src`.** Le paquet installé globalement (`pip install -e`) pointe vers une **autre copie** du projet (`Code\Chess\src`). Sans `PYTHONPATH=src`, c'est l'ancienne version qui tourne. Pour corriger durablement : `pip install -e .` depuis ce dépôt, dans un environnement où l'on a les droits d'écriture.
- **Moteur** : Stockfish 19, dans `tools/stockfish/stockfish.exe` (ignoré par git). Tout le corpus doit être analysé avec la même version.
- **Journaux** : dans `logs/`, un fichier par lancement.

## Décisions de méthode déjà prises (et pourquoi)

- **Pools** : on suit la classe de cadence enregistrée *au moment de la partie*.
  - chess.com classait le 10|0 en blitz jusqu'au 10/09/2020.
  - Lichess n'a de pool rapide que depuis le 01/12/2017. Son API ré-étiquette les vieilles parties selon les règles d'aujourd'hui : elles sont donc exclues avant 2018.
- **Contrôle de la cadence** : chaque modèle contrôle la durée estimée de la partie (base + 40 × incrément), car le mélange de cadences change d'une année à l'autre.
- **Conversion 2026** :
  - Rapide ↔ blitz sur un même site : on utilise les classements actuels des mêmes personnes.
  - chess.com ↔ Lichess : on utilise les comptes liés (profil Lichess qui déclare son compte chess.com).
  - Tables en « equipercentile » (même rang percentile), formule linéaire équivalente, filtrage robuste des valeurs aberrantes.
- **Échantillon chess.com rapide ↔ blitz** :
  - Au plus 12 personnes par compte « pivot ».
  - Autant de personnes trouvées via des parties blitz que via des parties rapides.
  - Sans cela, on sélectionne des joueurs sur leur classement rapide, et la conversion se décale d'environ 200 points en haut de l'échelle.
- **Références extérieures** : les dates des changements de règles et le comparatif ChessGoals (juillet 2026) sont sourcés dans le rapport.
