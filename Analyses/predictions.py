# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:08:32 2018

@author: Admin
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.stats as sms
from IPython.display import display, Markdown
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.stattools as sto
from pandas import DataFrame
import seaborn as sns
from testZ import *
sns.set_style("white")
#source: https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f


def get_orders(y,train_ratio):
    aic_min_order = []
    bic_min_order = []
    mse_min_order = []
    for tr in train_ratio:
        t = round(tr*len(y))
        y_tronq = y[:t]

        order_select = sto.arma_order_select_ic(y_tronq.values, ic = ['aic', 'bic'], max_ar=4, max_ma=4)
        aic_min_order.append(order_select["aic_min_order"])
        bic_min_order.append(order_select["bic_min_order"])
        mse_min_order.append(arma_select_mse(y_tronq.values,max_ar=4,max_ma=4))

    return aic_min_order,bic_min_order,mse_min_order

def get_values(train_ratio,aic_min_order,bic_min_order,mse_min_order,y,plot = False):
    aic_val = []
    bic_val=[]
    mse_val= []
    for i in range(3):
        tr = train_ratio[i]
        print('Train ratio :',tr)
        t = round(tr*len(y))
        y_tronq = y[:t]
        m,a,b = comparaison_model(aic_min_order[i],bic_min_order[i],mse_min_order[i],y_tronq,plot= plot)
        aic_val.append(a)
        bic_val.append(b)
        mse_val.append(m)
        
    return aic_val, bic_val, mse_val

def get_predictions(train_ratio,y,aic_min_order,bic_min_order,mse_min_order,plot=False,signif =False):
    err_aic = []
    err_bic = []
    err_mse = []
    for i in range(3):
        print('TRAIN RATIO :', train_ratio[i])
    
        print('MODEL AIC')
        (p,q) = aic_min_order[i]
        f,ci,e,s = out_of_sample_prediction(p=p, q=q, y_true=y, train_ratio=train_ratio[i],signif= signif)
        err_aic.append(e)
    
        print('MODEL BIC')
        (p,q) = bic_min_order[i]
        f,ci,e,s = out_of_sample_prediction(p=p, q=q, y_true=y, train_ratio=train_ratio[i],signif= signif)
        err_bic.append(e)
    
        print('MODEL MSE')
        (p,q) = mse_min_order[i]
        f,ci,e,s = out_of_sample_prediction(p=p, q=q, y_true=y, train_ratio=train_ratio[i],signif= signif)
        err_mse.append(e)
     
    if plot == True:
        plt.bar([0,1,2],[err_aic[0],err_bic[0],err_mse[0]])
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.25')
        plt.show()

        plt.bar([0,1,2],[err_aic[1],err_bic[1],err_mse[1]],color = 'g')
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.5')
        plt.show()

        plt.bar([0,1,2],[err_aic[2],err_bic[2],err_mse[2]], color = 'r')
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.75')
        plt.show()
    
    return err_aic,err_bic,err_mse 
    
def get_predictions2(train_ratio,z_,p_,aic_min_order,bic_min_order,mse_min_order,plot = False):
    err_aic = []
    err_bic = []
    err_mse = []
    for i in range(3):
        print('TRAIN RATIO :', train_ratio[i])
    
        print('MODEL AIC')
        (p,q) = aic_min_order[i]
        e = p_with_fit_of_z(p=p,q=q,z_true = z_ ,p_true =p_, train_ratio=train_ratio[i], signif = True)
        err_aic.append(e)
    
        print('MODEL BIC')
        (p,q) = bic_min_order[i]
        e = p_with_fit_of_z(p=p,q=q,z_true = z_ ,p_true =p_, train_ratio=train_ratio[i], signif = True)
        err_bic.append(e)
    
        print('MODEL MSE')
        (p,q) = mse_min_order[i]
        e = p_with_fit_of_z(p=p,q=q,z_true = z_ ,p_true =p_, train_ratio=train_ratio[i], signif = True)
        err_mse.append(e)
    
    if plot == True:
        plt.bar([0,1,2],[err_aic[0],err_bic[0],err_mse[0]])
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.25')
        plt.show()

        plt.bar([0,1,2],[err_aic[1],err_bic[1],err_mse[1]],color = 'g')
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.5')
        plt.show()

        plt.bar([0,1,2],[err_aic[2],err_bic[2],err_mse[2]], color = 'r')
        plt.xticks([0,1,2], ['aic_best_mod','bic_best_mod','mse_best_mod'])
        plt.title('MSE pred, train_ratio = 0.75')
        plt.show()
        
    return err_aic,err_bic,err_mse    
    

    