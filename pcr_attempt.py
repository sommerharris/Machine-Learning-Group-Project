
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib as plt

from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
def pcr_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for i in range(1, 40, 2):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")
        print(pls.score(X_test, y_test))


def pcr_cross_val(X, y):
    n = len(X)

    # 10-fold CV, with shuffle
    kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

    mse = []

    for i in np.arange(1, 20):
        pls = PLSRegression(n_components=i)
        score = model_selection.cross_val_score(pls, X, y, cv=kf_10
                                                ).mean()
        mse.append(-score)
    print(mse)

def l2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    l2_lin_model = Ridge(fit_intercept=0)
    l2_lin_model.fit(X_train, y_train)
    l2_predict = l2_lin_model.predict(X_test)

    l2_r2 = r2_score(y_pred=l2_predict, y_true=y_test)
    print("Ridge score regression")
    print(l2_r2)

def pca(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for i in range(1, 40, 1):
        pcr = make_pipeline(StandardScaler(), PCA(n_components=i), Ridge())
        pcr.fit(X_train, y_train)
        pca = pcr.named_steps["pca"]
        pca_predict = pcr.predict(X_test)
        pca_r2 = r2_score(y_pred = pca_predict, y_true= y_test)
        print(f"PCA score. Components:{i}")
        print(pca_r2)


def main():
    data = pd.read_csv('owid-covid-data.csv')
    uk_data = data.drop(data[data['iso_code'] != "GBR"].index)
    uk_data = uk_data.fillna(0)
    uk_data = uk_data.drop("iso_code", axis = 1)
    uk_data = uk_data.drop("continent", axis = 1)
    uk_data = uk_data.drop("location", axis = 1)

    num_cases = uk_data[['new_cases']].copy()
    uk_data = uk_data.drop("new_cases", axis = 1)
    uk_data = uk_data.drop("tests_units", axis = 1)

    uk_data['date'] = pd.to_datetime(uk_data['date'])
    uk_data['date'] = uk_data['date'].map(dt.datetime.toordinal)

#Need to drop columns related to new cases.
    uk_data = uk_data.drop("new_cases_smoothed", axis = 1)
    uk_data = uk_data.drop("new_cases_per_million", axis = 1)
    uk_data = uk_data.drop("new_cases_smoothed_per_million", axis = 1)

    pcr_regression(uk_data, num_cases)
    print(uk_data.columns)

if __name__=="__main__":
    main()