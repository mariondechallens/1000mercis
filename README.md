# 1000mercis

Descriptif succint des fichiers

### Preprocessing

+ `Data Import.ipynb` : chargement des données annonceur 1 et 2, formattage sous hdf avec réduction d'utilisation de mémoire et aggrégation sous format journalier.

### Visualisation
+ `Data visualization.ipynb` : visualisations des séries temporelles journalières (taux de conversion / taux de view) aggrégés par group de test (et par view).
+ Résultats exportés du notebook précédent sur le dataset annonceur 1 et 2.

### Analyses

#### Fichiers.py
+ `part1.py` : Graphiques de corrélation d'une série temporelle et d'analyse classique (ACF, PACF, QQplot, histogrammes).
+ `part2.py` : Graphiques de moyennes flottantes et des effets périodiques journaliers d'une série temporelle ainsi que sa décomposition (tendance, saison, cycle, bruit).
+ `part3.py` : Test de Dickey-Fuller et les différentes tranformations à appliquer à une série temporelle pour la rendre stationnaire (log, différencier).
+ `premiere_analyse.py` : Application des codes précédents aux campagnes 1000mercis après séparation en deux sous-datasets (version A et version B).
+ `testZ.py` :  Réalisation d'un test Z entre les versions A et B (pour leur taux de conversion) d'une bannière.

#### Notebooks des fonctions
+ `Visualisation d'une série temporelle et stationnarité (blog Datacay).ipynb` : Ensemble des fonctions présentes dans les fichiers `part1.py`, `part2.py` et `part3.py` pour l'analyse des taux de conversion comme séries temporelles.
+ `Analyse temporelle des taux de conversion.ipynb`: Fonctions du fichier `premiere_analyse.py` afin d'analyser les données de campagnes en tant que séries temporelles.

#### Notebooks des résultats
+ `Z test.ipynb` : Synthèse du test Z appliqué à toutes les campagnes.
+ `Annonceur1_campagne1_visite_2pages.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite 2 pages numéro 1 de l'annonceur 1.
+ `Annonceur1_campagne1_visite_engagee.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite engagée numéro 1 de l'annonceur 1.
+ `Annonceur1_campagne2_visite_2pages.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite 2 pages numéro 2 de l'annonceur 1.

+ `Annonceur2_campagne1_visite_achat.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite achat de l'annonceur 2
+ `Annonceur2_campagne1_visite_page_produit.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite page produit de l'annonceur 2.
+ `Annonceur2_campagne1_visite_panier.ipynb`: Résultats graphiques et numériques de l'analyse de la campagne visite panier de l'annonceur 2.
