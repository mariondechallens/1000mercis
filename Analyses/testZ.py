# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:08:32 2018

@author: Admin
"""

import pandas as pd
import numpy as np
from premiere_analyse import preparer
import scipy.stats as st

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
