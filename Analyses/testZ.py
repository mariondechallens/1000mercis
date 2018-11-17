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

n1 = len(dataA)   
n2 = len(dataB) 
s1 = sum(dataA['is_conv'])
s2 = sum(dataB['is_conv'])
p1 = s1/n1
p2 = s2/n2

Z = (p1-p2)/np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
#Z = (s1-s2)/np.sqrt(s1+s2)
Prej = 2*(1 - st.norm.cdf(Z))

fig, ax = plt.subplots(figsize=(12,6))
xA = np.linspace(s1-49, s1+50, 100)
yA = st.binom(n1, p1).pmf(xA)
ax.bar(xA, yA, alpha=0.5)
xB = np.linspace(s2-49, s2+50, 100)
yB = st.binom(n2, p2).pmf(xB)
ax.bar(xB, yB, alpha=0.5)
plt.xlabel('converted')
plt.ylabel('probability')
plt.title('Binomial distributions for the control A (red) and test B (blue) groups')


# standard error of the mean for both groups
SE_A = np.sqrt(p1 * (1-p1)) / np.sqrt(n1)
SE_B = np.sqrt(p2 * (1-p2)) / np.sqrt(n2)
# plot the null and alternate hypothesis
fig, ax = plt.subplots(figsize=(12,6))
x = np.linspace(0, .02, 1000)
yA = st.norm(p1, SE_A).pdf(x)
ax.plot(xA, yA)
ax.axvline(x=p1, c='red', alpha=0.5, linestyle='--')
yB = st.norm(p2, SE_B).pdf(x)
ax.plot(xB, yB)
ax.axvline(x=p2, c='blue', alpha=0.5, linestyle='--')
plt.xlabel('Converted Proportion')
plt.ylabel('PDF')
