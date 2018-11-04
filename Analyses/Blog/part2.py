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

#data = pd.read_csv('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/Codes/data_passengers.csv', header=0, index_col=0, parse_dates=True, sep=';')
#y = data['n_passengers']

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
    '''
    Plot rolling mean and rolling standard deviation for a given time series and window
    '''
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
#df = pd.read_csv('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/Codes/data_passengers.csv', header=0, index_col=0, parse_dates=True, sep=';')
def effet_mensuel(df):
    df['Month'] = df.index.strftime('%b')
    df['Year'] = df.index.year
 
# create nice axes names
    month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
 
# reshape data using 'Year' as index and 'Month' as column
    df_piv_line = df.pivot(index = 'Month', columns='Year', values='is_conv')
    df_piv_line = df_piv_line.reindex(index=month_names)
 
# create line plot
    df_piv_line.plot(colormap='jet')
    plt.title('Seasonal Effect per Month', fontsize=24)
    plt.ylabel('Taux moyen journalier')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.show()

# create new columns to DataFrame by extracting a string representing 
# the time under the control of an explicit format string
# '%b' extracts the month in locale's abbreviated name from the index
#df = pd.read_csv('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Projet/data_passengers.csv', header=0, index_col=0, parse_dates=True, sep=';')

def effet_mensuel2(df):
    df['Month'] = df.index.strftime('%b')
    df['Year'] = df.index.year
 
# create nice axes names
    month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
 
# reshape date
    df_piv_box = df.pivot(index='Year', columns='Month', values='is_conv')
 
# reindex pivot table with 'month_names'
    df_piv_box = df_piv_box.reindex(columns=month_names)
 
# create a box plot
    fig, ax = plt.subplots();
    df_piv_box.plot(ax=ax, kind='box');
    ax.set_title('Seasonal Effect per Month', fontsize=24);
    ax.set_xlabel('Month');
    ax.set_ylabel('Passengers');
    ax.xaxis.set_ticks_position('bottom');
    fig.tight_layout();
    plt.show()

# multiplicative seasonal decomposition
decomp = statsmodels.tsa.seasonal.seasonal_decompose(y, model='multiplicative')
decomp.plot();
plt.show()

decomp = statsmodels.tsa.seasonal.seasonal_decompose(y, model='additive')
decomp.plot();
plt.show()
