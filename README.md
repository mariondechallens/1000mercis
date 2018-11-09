# 1000mercis

Descriptif succint des fichiers

#### Preprocessing

+ `Data Import.ipynb` : chargement des données annonceur 1 et 2, formattage sous hdf avec réduction d'utilisation de mémoire.

#### Analyses

+ `part1.py` : Graphiques de corrélation d'une série temporelle et d'analyse classique (ACF, PACF, QQplot, historgrammes).
+ `part2.py` : Graphiques de moyennes flottantes et des effets périodiques journaliers d'une série temporelle ainsi que sa décomposition.
+ `part3.py` : Test de Dickey-Fuller et les différentes tranformations à appliquer à une série temporelle pour la rendre stationnaire (log, différencier).
+ `premiere_analyse.py` : Application des codes précédents aux campagnes 1000mercis après séparation en deux sous datasets (groupe A et groupeB).
