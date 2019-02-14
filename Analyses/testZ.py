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

sns.set_style("white")
#source: https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f


DATA_ALIAS = {
    'annonceur1/annonceur1_campaign1_visite_2pages.csv': "a1c1",
    'annonceur1/annonceur1_campaign2_visite_2pages.csv': "a1c2",
    'annonceur1/annonceur1_campaign3_visite_2pages.csv': "a1c3",
    'annonceur1/annonceur1_campaign4_visite_2pages.csv': "a1c4",
    'annonceur2/annonceur2_campaign1_achat.csv': "a2c1achat",
    'annonceur2/annonceur2_campaign1_visite_page_produit.csv': "a2c1produit",
    'annonceur2/annonceur2_campaign1_visite_panier.csv': "a2c1panier",
}

def proportions(data):
    dataA = data.loc[data['group']=="A",:]
    dataB = data.loc[data['group']=="B",:]

    n1 = len(dataA)   
    n2 = len(dataB) 
    s1 = sum(dataA['is_conv'])
    s2 = sum(dataB['is_conv'])
    p1 = s1/n1
    p2 = s2/n2
    return dataA, dataB, n1, n2, s1, s2, p1, p2
    
def testZ(data):
    dataA, dataB, n1, n2, s1, s2, p1, p2 = proportions(data)

    Z = (p1-p2)/np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    #Z = (s1-s2)/np.sqrt(s1+s2)
    Prej = 2*(1 - st.norm.cdf(abs(Z)))

    #test = sms.weightstats.ztest(dataA['is_conv'],dataB['is_conv']) #ok mêmes valeurs obtenues
    
    return Z, Prej


def testZ_cum(data):
    """
    Réalisation du test Z sur 2 groupes A et B de manière cumulée (du jour 1 au jour 2, 3, ...T)
    """
    if "date" not in data.columns:
        data.loc[:, "date"] = pd.to_datetime(data["impression_date"], format="%Y-%m-%d %H:%M:%S")
    # comptage du nombre de donnees par groupe
    daily_size = data.groupby(["date", "group", "is_conv"]).size()
    daily_size = daily_size.rename('n').reset_index()
    # nombre total cumule par groupe
    n_cum = daily_size.groupby(['date', 'group'])['n'].sum().unstack().cumsum()
    # nombre de succes (=1) par groupe
    s_cum = daily_size.loc[daily_size["is_conv"] == 1].groupby(['date', 'group'])['n'].sum().unstack()
    s_cum = s_cum.reindex(n_cum.index).fillna(0).cumsum()  # NaN quand il y a pas de conversion
    p_cum = s_cum / n_cum
    Z_cum = (p_cum["A"] - p_cum["B"]) / np.sqrt((p_cum * (1 - p_cum) / n_cum).sum(1))
    P_rej = pd.Series(2 * (1 - st.norm.cdf(Z_cum.abs())), index=p_cum.index, name='P_rej')
    return Z_cum, P_rej, p_cum


def testZ_cum_frequency(data, freq="1D"):
    """
    Realise test Z cumule a une frequence donnee. Par exemple, si la frequence est de 3D ie 3 jours,
    on fait un test Z avec les donnes de t0 a t2, puis de t0 a t5, puis t0 a t8 etc.
    """
    if "date" not in data.columns:
        data.loc[:, "date"] = pd.to_datetime(data["impression_date"], format="%Y-%m-%d %H:%M:%S")
    if pd.Timedelta(freq) > pd.Timedelta("1D"):
        data_grouped = data.set_index(data["date"].dt.normalize()).groupby("group")
    else:
        data_grouped = data.set_index("date").groupby("group")
    n = data_grouped.resample(freq, closed="left", label="right").size().T
    index = (n > 0).any(axis=1)
    n_cum = n.loc[index].cumsum()
    s_cum = data_grouped.resample(freq, closed="left", label="right")['is_conv'].sum().unstack(0)
    s_cum = s_cum.loc[index].cumsum()
    p_cum = s_cum / n_cum
    Z_cum = (p_cum["A"] - p_cum["B"]) / np.sqrt((p_cum * (1 - p_cum) / n_cum).sum(1))
    P_rej = pd.Series(2 * (1 - st.norm.cdf(Z_cum.abs())), index=p_cum.index, name='P_rej')
    return Z_cum, P_rej, p_cum


def plot_testZ_cum_frequency(Z_cum, P_rej, p_cum, freq):
    """
    Resultats du test Z cumule a la frequence donnee.
    """
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    p_cum.plot(
        title="Evolution du taux de conversion cumule",
        ax=plt.gca(), marker="o", ms=4
    )
    plt.ylabel("Taux de conversion")

    plt.subplot(1, 3, 2)
    P_rej.plot(
        title='Evolution de la P-valeur cumulé',
        ax=plt.gca(), label="P-val", marker="o", ms=4
    )
    for threshold in [0.2, 0.1]:
        theshold_series = pd.Series(np.full(len(P_rej), threshold), index=P_rej.index)
        theshold_series.plot(ax=plt.gca(), label=f"threshold={threshold}")
    plt.ylabel('P-val cumulé')
    plt.legend()

    plt.subplot(1, 3, 3)
    Z_cum.plot(title='Evolution de la Z-valeur cumulé', ax=plt.gca(), marker="o", ms=4)
    plt.ylabel('Z-val cumulé')

    plt.suptitle(f"Z-test cumulé avec pas = {freq}")
    plt.show();


#tracé de la distribution binomiale des taux de conversion
def binom_distri(data):
    dataA, dataB, n1, n2, s1, s2, p1, p2 = proportions(data)
    
    fig, ax = plt.subplots(figsize=(12,6))
    xA = np.linspace(s1-49, s1+50, 100)
    yA = st.binom(n1, p1).pmf(xA)
    ax.bar(xA, yA, alpha=0.5)
    xB = np.linspace(s2-49, s2+50, 100)
    yB = st.binom(n2, p2).pmf(xB)
    ax.bar(xB, yB, alpha=0.5)
    plt.xlabel('Conversion eut lieu')
    plt.ylabel('Probabilité')
    plt.title('Distribution binomiale pour le groupe contrôle A (rouge) and le test B (bleu)')

#tracé de la distribution normale sous Ho des taux de conversion
def norm_distri(data):
    dataA, dataB, n1, n2, s1, s2, p1, p2 = proportions(data)
    # standard error of the mean for both groups
    SE_A = np.sqrt(p1 * (1-p1)) / np.sqrt(n1)
    SE_B = np.sqrt(p2 * (1-p2)) / np.sqrt(n2)
    # plot the null and alternate hypothesis
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.linspace(0.006, .01, 1000)
    yA = st.norm(p1, SE_A).pdf(x)
    ax.plot(x, yA,c='red')
    ax.axvline(x=p1, c='red', alpha=0.5, linestyle='--')
    yB = st.norm(p2, SE_B).pdf(x)
    ax.plot(x, yB,c='blue')
    ax.axvline(x=p2, c='blue', alpha=0.5, linestyle='--')
    plt.xlabel('Taux de conversion')
    plt.ylabel('Densité de probabilité')
    plt.title('Distribution normale pour le groupe contrôle A (rouge) and le test B (bleu)')

def testARMA(s,p,d,q):  #ttrouver un modèle ARIMA pour les données (d = ordre différenciation)
    display(Markdown("## Fit du modèle"))
    print('\n')
    model = ARIMA(s, order=(p,d,q))  
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    display(Markdown("## Erreurs"))
    print('\n')
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())


def arma_model_selection(series, max_ar=4, max_ma=4):
    assert not series.isnull().any()
    order_select = sto.arma_order_select_ic(
        series.values,
        ic=['aic', 'bic'],
        max_ar=max_ar,
        max_ma=max_ma
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(order_select["aic"])
    plt.xlabel("Ordre MA")
    plt.ylabel("Ordre AR")
    plt.title("Résultats AIC")

    plt.subplot(1, 2, 2)
    sns.heatmap(order_select["bic"])
    plt.xlabel("Ordre MA")
    plt.ylabel("Ordre AR")
    plt.title("Résultats BIC")

    plt.suptitle(f"max_ar={max_ar}, max_ma={max_ma}")
    plt.show();

    aic_min_order = order_select["aic_min_order"]
    bic_min_order = order_select["bic_min_order"]
    print(
        "AIC meilleur modèle : AR={}, MA={}, AIC={} ".format(
            aic_min_order[0], aic_min_order[1],
            order_select['aic'].loc[aic_min_order]
        )
    )
    print(
        "BIC meilleur modèle : AR={}, MA={}, BIC={} ".format(
            bic_min_order[0], bic_min_order[1],
            order_select['bic'].loc[bic_min_order]
        )
    )

    return order_select


def arma_summary(p, q, y_true):
    model = ARIMA(y_true, order=(p, 0, q)).fit()
    print(model.summary())
    # print(model_fit.summary())
    display(Markdown("## Erreurs sur la période d'entraînement"))
    print('\n')
    residuals = DataFrame(model.resid)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    residuals.plot(ax=plt.gca())
    plt.title("Evolution de l'erreur")

    plt.subplot(1, 2, 2)
    residuals.plot(kind='kde', ax=plt.gca())
    plt.title("Répartition de l'erreur")

    plt.show()


def in_sample_prediction(p, q, y_true, train_ratio):
    if isinstance(y_true, pd.Series):
        # on oublie les dates et on regarde que si la position du jour par rapport au debut du test
        y_true = y_true.values
        # cela permet d'eviter des problemes comme par ex. l'existence d'un seul NaN au milieu des donnees

    t = round(train_ratio * len(y_true))
    model = ARIMA(y_true, order=(p, 0, q)).fit()  # on fit avec toutes les donnees

    train_data = y_true[:t]  # on ne fit pas le modele dessus

    pred_start = t
    pred_end = len(y_true)

    pred_index = np.arange(pred_start + 1, pred_end + 1)
    dynamic_predictions = model.predict(start=pred_start, end=pred_end - 1, dynamic=True)
    one_step_ahead_predictions = model.predict(start=pred_start, end=pred_end - 1, dynamic=False)

    plt.figure(figsize=(16, 4))
    plt.plot(np.arange(1, t + 1), train_data, label="Observed (train)", marker="o", ms=4)
    plt.plot(pred_index, y_true[t:], label="Test period (truth)", marker="o", ms=4)
    plt.plot(pred_index, dynamic_predictions, label="Dynamic pred", marker="o", ms=4)
    plt.plot(pred_index, one_step_ahead_predictions, label="1-step pred", marker="o", ms=4)
    plt.legend()
    plt.title(f"[train_ratio={train_ratio}] Resultats de prédiction pour AR={p} MA={q}")
    plt.xlabel("Jour du test")
    plt.xticks()
    plt.show()


def out_of_sample_prediction(p, q, y_true, train_ratio, signif = True):
    t = round(train_ratio * len(y_true))
    train_data = y_true[:t]

    model = ARIMA(train_data, order=(p, 0, q)).fit()

    pred_start = t
    pred_end = len(y_true)

    pred_index = np.arange(pred_start + 1, pred_end + 1)
    dynamic_predictions = model.predict(start=pred_start, end=pred_end - 1, dynamic=True)
    one_step_ahead_predictions = model.predict(start=pred_start, end=pred_end - 1, dynamic=False)
    # les predictions dynamiques et one step ahead doivent etre identiques
    assert all(one_step_ahead_predictions == dynamic_predictions)

    plt.figure(figsize=(16, 4))
    plt.plot(np.arange(1, t + 1), train_data, label="Observed (train)", marker="o", ms=4)
    plt.plot(pred_index, y_true[t:], label="Test period (truth)", marker="o", ms=4)
    plt.plot(pred_index, dynamic_predictions, label="Dynamic pred", marker="o", ms=4)
    plt.plot(pred_index, one_step_ahead_predictions, label="1-step pred", marker="o", ms=4)

    plt.legend()
    plt.title(f"[train_ratio={train_ratio}] Resultats de prédiction pour AR={p} MA={q}")
    
    if signif == True:
        for threshold in [0.2, 0.1]:
            threshold_series = pd.Series(np.full(len(y_true), threshold))
            plt.plot(threshold_series, label=f"threshold={threshold}")
  
    plt.legend()
    plt.show()
