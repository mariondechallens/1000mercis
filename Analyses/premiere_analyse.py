# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:03:36 2018

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
#parties du blog
from part1 import *
from part2 import *
from part3 import *
#import dateutil

#import data
folder = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/Données/'
annonceur = 'annonceur1/annonceur1'
campagne = 'annonceur1_campaign1_visite_engagee'
data = pd.read_hdf(folder + annonceur + '.hdf', key=campagne)


#convert data times to date times
#data['impression_date'] = data['impression_date'].apply(dateutil.parser.parse, dayfirst=True)


#exploration des données
def explorer(data):
    print('Début')
    print(data.head()) 
    print('Taille')
    print(data.shape)
    print('Colonnes')
    print(data.columns)
    print('Types')
    print(data.dtypes)
    print('Infos')
    print(data.info())
    print('Description')
    print(data.describe())
    
def preparer(data):
    print('Conversion des index en dates')
    data['impression_date'] = pd.to_datetime(data['impression_date'])
    
    print('Moyennes des taux par jour et séparation en deux groupes A et B')
    dataA = data.loc[data['group']=="A",:]
    dataA = dataA.groupby(by = dataA['impression_date'].dt.date)[['view','is_conv']].mean()

    dataB = data.loc[data['group']=="B",:]
    dataB = dataB.groupby(by = dataB['impression_date'].dt.date)[['view','is_conv']].mean()

"""
dataA['is_conv'].value_counts().plot.pie()
dataA['view'].value_counts().plot.pie()

dataB['is_conv'].value_counts().plot.pie()
dataB['view'].value_counts().plot.pie() """

#histogramme 
data.hist(column='is_conv',by='group')

#comparaison des distributions avec un boxplot 
data.boxplot(column='is_conv',by='group')

#diagramme à secteurs
data['group'].value_counts().plot.pie()

def analyser(data):
    print('Histogramme')
    data.hist(column='is_conv')
    
    print('Scatter plot')
    data.plot.scatter(x='is_conv',y='view')
    
    print('Densité du taux de conversion')
    data['is_conv'].plot.kde()
#part 1 du blog
    print('Corrélations et covariances')
    y = data['is_conv']
    y.index = pd.to_datetime(y.index)
    corr(y) ## scatter plots pour la corrélation
    ts_plot(y) ##analyse classique d'une ST (ACF, PACF, QQ et histo)

#part 2 blog
    print('Moyennes flottantes')
    rolling_mean(y) #regarder la moyenne flottante et l'écart type
    plot_rolling_average(y)
    
    print('Effet journalier')
    y2 = pd.Series.to_frame(y)
    effet_journalier(y2) #regarder par jour


# multiplicative and additive seasonal decomposition

##problème de fréquence !!
    print('Décomposition de la série de temps')
    decomp = seasonal_decompose(y, model='multiplicative',freq=20)
    decomp.plot();
    plt.show()

    decomp = seasonal_decompose(y, model='additive',freq = 30)
    decomp.plot();
    plt.show()

### Part 3: test de Dickey-Fuller
    print('Test de Dickey-Fuller')
    adf_test(y)
    ts_diagnostics(y)

# difference time series
    print('Stationnariser la série')
    print('Différencier')
    y_diff = np.diff(y)
    ts_diagnostics(y_diff, lags=30)

# log transform time series
    print('Passer au logarithme')
    y_log = np.log(y)
    ts_diagnostics(y_log, lags=30)

 #log difference time series
    print('Différencier le logarithme')
    y_log_diff = np.log(y).diff().dropna()
    ts_diagnostics(y_log_diff, lags=30)

# log difference time series *2
    print('Différencier le logarithme deux fois')
    y_log_diff2 = np.log(y).diff().diff(12).dropna()
    ts_diagnostics(y_log_diff2, lags=30)
    
analyser(dataA)
analyser(dataB)