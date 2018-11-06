# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:23:39 2018

@author: Admin
"""

# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.tsa.api as smt
import statsmodels.api as sm

#data = pd.read_csv('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/data_passengers.csv', header=0, index_col=0, parse_dates=True, sep=';')

# create Series object
#y = data['n_passengers']

def adf_test(y):
    # perform Augmented Dickey Fuller test
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
 
# apply the function to the time series
#adf_test(y)

def ts_diagnostics(y, lags=None, title='Taux de conversion'):
    '''
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    # weekly moving averages (5 day window because of workdays)
    rolling_mean = y.rolling(window=12).mean()
    rolling_std = y.rolling(window=12).std()
    
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))
    
    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson');
    rolling_std.plot(ax=ts_ax, color='darkslateblue');
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24);
    
    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5) 
    
    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    
    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    plt.show()
    
    # perform Augmented Dickey Fuller test
    print('Results of Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return

"""
# difference time series
y_diff = np.diff(y)
 
# compute time series diagnostics
ts_diagnostics(y_diff, lags=30)
adf_test(y_diff)


# log transform time series
y_log = np.log(y)
 
# compute time series diagnostics
ts_diagnostics(y_log, lags=30)
adf_test(y_log)

 #log difference time series
y_log_diff = np.log(y).diff().dropna()
 
# compute time series diagnostics
ts_diagnostics(y_log_diff, lags=30)
adf_test(y_log_diff)

# log difference time series *2
y_log_diff2 = np.log(y).diff().diff(12).dropna()
 
# compute time series diagnostics
ts_diagnostics(y_log_diff2, lags=30)
adf_test(y_log_diff2)
"""
