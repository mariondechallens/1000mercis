# 1000mercis

Descriptif succint des fichiers

#### Preprocessing

+ `Data Import.ipynb` : chargement des données annonceur 1 et 2, formattage sous hdf avec réduction d'utilisation de mémoire et aggrégation sous format journalier.

#### Visualisation
+ `Data visualization.ipynb` : visualisations des séries temporelles journalières (taux de conversion / taux de view) aggrégés par group de test (et par view).
+ Résultats exportés du notebook précédent sur le dataset annonceur 1 et 2.

#### Analyses

+ `part1.py` : Graphiques de corrélation d'une série temporelle et d'analyse classique (ACF, PACF, QQplot, histogrammes).
+ `part2.py` : Graphiques de moyennes flottantes et des effets périodiques journaliers d'une série temporelle ainsi que sa décomposition (tendance, saison, cycle, bruit).
+ `part3.py` : Test de Dickey-Fuller et les différentes tranformations à appliquer à une série temporelle pour la rendre stationnaire (log, différencier).
+ `premiere_analyse.py` : Application des codes précédents aux campagnes 1000mercis après séparation en deux sous-datasets (version A et version B).
