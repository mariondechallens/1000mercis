# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:46:12 2018

@author: Admin
"""

# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa 

# simple line plot
def rolling_mean(y):
    plt.plot(y)
    plt.title('Taux de conversion', fontsize=24)
    plt.ylabel('taux moyen journalier')
    plt.xlabel('Date')
    plt.show()

    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False);
    fig.set_figwidth(14);
    fig.set_figheight(8);
 
# push data to each ax
#upper left
    axes[0][0].plot(y.index, y, label='Original');
    axes[0][0].plot(y.index, y.rolling(window=4).mean(), label='4-Months Rolling Mean', color='crimson');
    axes[0][0].set_xlabel("Date");
    axes[0][0].set_ylabel("taux moyen journalier");
    axes[0][0].set_title("4-Months Moving Average");
    axes[0][0].legend(loc='best');
 
# upper right
    axes[0][1].plot(y.index, y, label='Original')
    axes[0][1].plot(y.index, y.rolling(window=6).mean(), label='6-Months Rolling Mean', color='crimson');
    axes[0][1].set_xlabel("Date");
    axes[0][1].set_ylabel("taux moyen journalier");
    axes[0][1].set_title("6-Months Moving Average");
    axes[0][1].legend(loc='best');
 
# lower left
    axes[1][0].plot(y.index, y, label='Original');
    axes[1][0].plot(y.index, y.rolling(window=8).mean(), label='8-Months Rolling Mean', color='crimson');
    axes[1][0].set_xlabel("Date");
    axes[1][0].set_ylabel("taux moyen journalier");
    axes[1][0].set_title("8-Months Moving Average");
    axes[1][0].legend(loc='best');
 
# lower right
    axes[1][1].plot(y.index, y, label='Original');
    axes[1][1].plot(y.index, y.rolling(window=12).mean(), label='12-Months Rolling Mean', color='crimson');
    axes[1][1].set_xlabel("Date");
    axes[1][1].set_ylabel("taux moyen journalier");
    axes[1][1].set_title("12-Months Moving Average");
    axes[1][1].legend(loc='best');
    plt.tight_layout();
    
    
def plot_rolling_average(y, window=12):
    """
    Plot rolling mean and rolling standard deviation for a given time series and window
    """
    # calculate moving averages
    rolling_mean = y.rolling(window=window).mean()
    rolling_std = y.rolling(window=window).std()
 
    # plot statistics
    plt.plot(y, label='Original')
    plt.plot(rolling_mean, color='crimson', label='Moving average mean')
    plt.plot(rolling_std, color='darkslateblue', label='Moving average standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    return    
# create new columns to DataFrame by extracting a string representing 
# the time under the control of an explicit format string
# '%b' extracts the month in locale's abbreviated name from the index


def effet_journalier(df):
    """
    Influence du jour du mois (1,...31) sur le taux de conversion moyen
    """
    df['Day'] = df.index.day
    df['Month'] = df.index.strftime('%b')
    
    # reshape data pour plot
    df_piv_line = df.pivot(index='Day', columns='Month', values='is_conv')
 
    # create line plot
    df_piv_line.plot(colormap='jet')
    plt.title('Seasonal Effect per Day', fontsize=24)
    plt.ylabel('Taux moyen journalier')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.show()

    # reshape date pour boxplot
    df_piv_box = df.pivot(index='Month', columns='Day', values='is_conv')

    # create a box plot
    fig, ax = plt.subplots();
    df_piv_box.plot(ax=ax, kind='box');
    ax.set_title('Seasonal Effect per Day', fontsize=24);
    ax.set_xlabel('Day');
    ax.set_ylabel('Taux moyen journalier');
    ax.xaxis.set_ticks_position('bottom');
    fig.tight_layout();
    plt.show()

# multiplicative seasonal decomposition


