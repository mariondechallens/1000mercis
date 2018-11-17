# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:08:32 2018

@author: Admin
"""

import pandas as pd
import numpy as np
from premiere_analyse import preparer
import scipy.stats as st
import matplotlib.pyplot as plt
#import data
folder = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/Donnees/'
annonceur = 'annonceur1/annonceur1'
campagne = 'annonceur1_campaign1_visite_engagee'
data = pd.read_hdf(folder + annonceur + '.hdf', key=campagne)

#sur données brutes  (pas la moyenne)
dataA = data.loc[data['group']=="A",:]
dataB = data.loc[data['group']=="B",:]
    
s1 = sum(dataA['is_conv'])
s2 = sum(dataB['is_conv'])
p1 = s1/len(dataA)
p2 = s2/len(dataB)

Z = (s1-s2)/np.sqrt(s1+s2)
Prej = 2*(1 - st.norm.cdf(Z))

fig, ax = plt.subplots(figsize=(12,6))
xA = np.linspace(s1-49, s1+50, 100)
yA = st.binom(len(dataA), p1).pmf(xA)
ax.bar(xA, yA, alpha=0.5)
xB = np.linspace(s2-49, s2+50, 100)
yB = st.binom(len(dataB), p2).pmf(xB)
ax.bar(xB, yB, alpha=0.5)
plt.xlabel('converted')
plt.ylabel('probability')
plt.title('Binomial distributions for the control (red) and test (blue) groups')