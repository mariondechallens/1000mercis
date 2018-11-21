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

#import data
folder = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/Donnees/'
annonceur = 'annonceur1/annonceur1'
campagne = 'annonceur1_campaign1_visite_engagee'
data = pd.read_hdf(folder + annonceur + '.hdf', key=campagne)

#convert data times to date times
#data['impression_date'] = data['impression_date'].apply(dateutil.parser.parse, dayfirst=True)

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
    
    print('Histogramme par groupe') 
    data.hist(column='is_conv',by='group')
    
    print('Boxplot par groupe')
    data.boxplot(column='is_conv',by='group')
    
    print('Répartition des groupes')
    data['group'].value_counts().plot.pie()
    
def preparer(data):
    print('Conversion des index en dates')
    data['impression_date'] = pd.to_datetime(data['impression_date'])
    
    print('Moyennes des taux par jour et séparation en deux groupes A et B')
    dataA = data.loc[data['group']=="A",:]
    dataA = dataA.groupby(by = dataA['impression_date'].dt.date)[['view','is_conv']].mean()

    dataB = data.loc[data['group']=="B",:]
    dataB = dataB.groupby(by = dataB['impression_date'].dt.date)[['view','is_conv']].mean()
    
    return dataA, dataB
"""
dataA['is_conv'].value_counts().plot.pie()
dataA['view'].value_counts().plot.pie()

dataB['is_conv'].value_counts().plot.pie()
dataB['view'].value_counts().plot.pie() """



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

"""
#part 2 blog
    print('Moyennes flottantes')
    rolling_mean(y) #regarder la moyenne flottante et l'écart type
    plot_rolling_average(y) """
    
    print('Effet journalier')
    y2 = pd.Series.to_frame(y)
    effet_journalier(y2) #regarder par jour


# multiplicative and additive seasonal decomposition

##problème de fréquence !!
    print('Décomposition de la série de temps selon modèle multiplicatif')
    decomp = seasonal_decompose(y, model='multiplicative',freq=1)
    decomp.plot();
    plt.show()

"""
    decomp = seasonal_decompose(y, model='additive',freq = 1)
    decomp.plot();
    plt.show()"""

### Part 3: test de Dickey-Fuller
    print('Test de Dickey-Fuller')
    adf_test(y)
    ts_diagnostics(y)

def transformer(data,transfo): #transfo = diff1, log, logdiff1,logdiff2
    y = data['is_conv']
    if transfo == "diff1":
        print("Différencier à l'ordre 1: y_t - y_[t-1])
        y_tr = np.diff(y)
    
    if transfo == "log":
        print('Passer au logarithme')
        y_tr = np.log(y)
        
    if transfo == "logdiff":
        print('Différencier le logarithme')
        y_tr = np.log(y).diff().dropna()
    
    if transfo == "logdiff2":
        print('Différencier le logarithme deux fois')
        y_tr = np.log(y).diff().diff(12).dropna()
        
    
    print("Analyse après transformation")
    ts_diagnostics(y_tr, lags=30)


"""
dataA, dataB = preparer(data) 
explorer(data)  
analyser(dataA)
analyser(dataB)"""
