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
data['impression_date'] = pd.to_datetime(data['impression_date'])

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
    

#moyenne des taux de conversion par jour
     
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
dataA.hist(column='is_conv')
data.hist(column='is_conv',by='group')

#comparaison des distributions avec un boxplot 
data.boxplot(column='is_conv',by='group')

#scatterplot :
dataA.plot.scatter(x='is_conv',y='view')
 #density plot 
dataA['is_conv'].plot.kde()

#diagramme à secteurs
data['group'].value_counts().plot.pie()

y = dataA['is_conv']
y.index = pd.to_datetime(y.index)
corr(y) ## scatter plots pour la corrélation
ts_plot(y) ##analyse classique d'une ST (ACF, PACF, QQ et histo)

#part 2 blog
rolling_mean(y) #regarder la moyenne flottante et l'écart type
plot_rolling_average(y)

y2 = pd.Series.to_frame(y)
effet_journalier(y2) #regarder par jour


# multiplicative and additive seasonal decomposition

##problème de fréquence !!
decomp = seasonal_decompose(y, model='multiplicative',freq = 2)
decomp.plot();
plt.show()

decomp = seasonal_decompose(y, model='additive',freq = 30)
decomp.plot();
plt.show()

### Part 3: test de Dickey-Fuller
adf_test(y)
ts_diagnostics(y)
