# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 19:11:15 2018

@author: Admin
"""



# required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import seaborn as sns


def corr(data) :
    # build scatterplot
    ncols = 3
    nrows = 3
    lags = 9
 
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 6 * nrows))
 
    for ax, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
        lag_str = 't-{}'.format(lag)
        X = (pd.concat([data, data.shift(-lag)], axis=1, keys=['y']+[lag_str]).dropna())
    
    # plot data
        X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
        corr = X.corr().as_matrix()[0][1]
        ax.set_ylabel('Original');
        ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
        ax.set_aspect('equal');
    
        # top and right spine from plot
        sns.despine();
 
    fig.tight_layout()
    plt.show()


def ts_plot(y, lags=None, title=''):
    """
    Calculate acf, pacf, histogram, and qq plot for a given time series
    """
    # if time series is not a Series object, make it so
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
    
    # initialize figure and axes
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))
    
    # time series plot
    y.plot(ax=ts_ax)
    ts_ax.legend(loc='best')  # plt.legend(loc='best')
    ts_ax.set_title(title);
    
    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('Normal QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    plt.show()
    return


""" 
np.random.seed(1)
# simulate discrete Gaussian white noise N(0, 1)
e = np.random.normal(size=1000)
#e = np.random.standard_t(size=1000, df=1)

ts_plot(e, lags=30, title='white_noise') 
"""
