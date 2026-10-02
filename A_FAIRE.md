# Ce qu'il reste à faire

État au 2 octobre 2026, 3h10. L'étude 2014-2026 (chess.com + Lichess, conversion
2026 entre les quatre pools) est terminée et commitée.

## Où on en est

| Étape | État | Détail |
|---|---|---|
| Collecte chess.com 2014–2026 | ✅ terminée | 10 414 parties. |
| Collecte Lichess 2014–2026 | ✅ terminée | 6 957 parties. |
| Analyse moteur | ✅ terminée | 20 512 parties sur 20 517 analysées. Les 5 manquantes sont des tournois thématiques à position imposée, exclus pour toujours (le moteur ne peut pas les scorer) : `evaluate` ne progressera plus dessus, ne pas s'inquiéter si elles réapparaissent dans un futur `evaluate --workers N`, il suffit de l'arrêter après quelques minutes. |
| Sondage de conversion | ✅ terminé | 8 486 comptes chess.com, 23 518 comptes Lichess, 611 comptes liés. |
| Rapport final | ✅ généré | `reports/findings.md` (+ `reports/tables/*.csv`, `reports/figures/*.png`) sur le corpus complet, remplace l'ancien brouillon 2018-19/2024-25. |
| Commit | ✅ fait | Tout le travail en attente a été découpé en commits logiques (client HTTP partagé, collecte Lichess, sondage de conversion, comparaisons année/site, rapport, docs, rapport final). |

## Points relus dans le rapport

- **Conversion chess.com rapide ↔ blitz** : la relation bend est signalée (« max bend 100 »), le rapport recommande explicitly la table plutôt que la formule pour cette paire.
- **chess.com / Lichess par année** : tourne maintenant sur les vraies données des deux sites, y compris le rapide (qui manquait dans le brouillon précédent). Le rapport signale lui-même que cette estimation ne s'accorde pas avec la conversion par comptes liés (écart médian -305 points) et explique pourquoi (pente accuracy→rating plate), en recommandant d'utiliser la conversion par comptes liés pour convertir une note.
- **Rupture de 2020 sur chess.com rapide** : la section *Year by year* présente bien les tendances avant/après séparément.
- **Figures** : les 5 PNG dans `reports/figures/` sont générés et non vides (87 Ko à 540 Ko).

## Reste en suspens, mineur

- `CLAUDE.md` à la racine existe mais est vide — non commité, à remplir ou supprimer si inutile.
- `prof` à la racine est un fichier binaire de 106 Ko non suivi par git, sans rapport apparent avec le pipeline (probablement une sortie de profiling oubliée) — à vérifier et supprimer si inutile.
- Pas d'autre tâche identifiée : relire `reports/findings.md` dans son ensemble si une publication est prévue.

## À savoir pour reprendre

- **Lancer avec `PYTHONPATH=src`.** Le paquet installé globalement (`pip install -e`) pointe vers une **autre copie** du projet (`Code\Chess\src`). Sans `PYTHONPATH=src`, c'est l'ancienne version qui tourne.
- **Moteur** : Stockfish 19, dans `tools/stockfish/stockfish.exe` (ignoré par git). Tout le corpus a été analysé avec cette version.
- **Journaux** : dans `logs/`, un fichier par lancement.
- **Lichess** : ne jamais lancer deux collectes en même temps (une seule requête à la fois ; un flux abandonné bloque l'IP environ 1h).

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
