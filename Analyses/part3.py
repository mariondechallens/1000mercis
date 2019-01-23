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
from IPython.display import display, Markdown


def adf_test(y):
    """
    Perform Augmented Dickey Fuller test
    """
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    display(dfoutput.rename('Results of Augmented Dickey-Fuller test:').to_frame())


"""
# apply the function to the time series
adf_test(y)
"""


def ts_diagnostics(y, lags=None, title='Taux de conversion', window=5):
    """
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
        
    # weekly moving averages (5 day window because of workdays)
    rolling_mean = y.rolling(window=window).mean()
    rolling_std = y.rolling(window=window).std()
    
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))
    
    # time series plot
    y.plot(ax=ts_ax, label=y.name)
    rolling_mean.plot(ax=ts_ax, color='crimson', label=f'roll_mean ({window} days)');
    rolling_std.plot(ax=ts_ax, color='darkslateblue', label=f'roll_std ({window} days)');
    ts_ax.legend(loc='best')
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
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    display(dfoutput.rename('Results of Augmented Dickey-Fuller test:').to_frame())
    return


