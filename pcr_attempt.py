
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
def pls_regression(X, y):


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
    print(y_train.count())
    print(y_test.count())
    score = []
    MSE = []
    coefficients = []

    for i in range(1, 40, 1):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")

        score.append(pls.score(X_test, y_test))
        y_pred = pls.predict(X_test)
        MSE.append(mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False))

        coefficients.append([])

        for j in pls.coef_:
            coefficients[i-1].append(j[0])

        print(f"MSE: {MSE[i // 3]}")
        print(score[i-1])
        print(pls.coef_)

    # plt.plot(list(range(1, 40, 1)), score)
    # plt.title("PLS regression")
    # plt.xticks(list(range(0, 41, 2)))
    # plt.xlabel("Number of components")
    # plt.ylabel("Score (R^2)?")
    # plt.title("Score with varying number of components")
    # plt.show()



    # plt.plot(list(range(1, 40, 1)), MSE)
    # plt.xticks(list(range(0, 41, 2)))
    # plt.xlabel("Number of components")
    # plt.ylabel("MSE")
    # plt.title("Score with varying number of components")
    # # plt.show()


    return score, MSE
    # So for this part, I actually wanted to see that the pls model performed dimension reduction
    # but is seems like there's still just as many dimensions. I know that the column headings that I
    # added don't actually correspond to the data shown but we should at least see more 0's.
    coefficients = pd.DataFrame(coefficients, columns=X_train.columns)
    print(coefficients)

    '''
    plt.plot(list(range(1, 40, 1)), MSE)
    plt.xticks(list(range(0, 41, 2)))
    plt.xlabel("Number of components")
    plt.ylabel("MSE")
    plt.title("Score with varying number of components")
    plt.show()
    '''




def pls_cross_val(X, y):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.2)
    score = []
    MSE = []
    coefficients = []

    for i in range(1, 40, 1):
        pcr = make_pipeline(StandardScaler(), PCA(n_components=i), Ridge())
        pcr.fit(X_train, y_train)
        pca = pcr.named_steps["pca"]
        y_pred = pcr.predict(X_test)
        '''
        pca_r2 = r2_score(y_pred = pca_predict, y_true= y_test)
        print(f"PCA score. Components:{i}")
        '''
        MSE.append(mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False))
        score.append(pcr.score(X_test, y_test))
        coefficients.append([])

        # for j in pcr.coef_:
        #     coefficients[i-1].append(j[0])

        print(f"MSE: {MSE[i // 3]}")
        print(score[i-1])
        # print(pcr.coef_)

    # plt.plot(list(range(1, 40, 1)), score)
    # plt.xticks(list(range(0, 41, 2)))
    # plt.xlabel("Number of components")
    # plt.ylabel("Score (R^2)?")
    # plt.title("Score with varying number of components")
    # # plt.show()
    #
    # plt.plot(list(range(1, 40, 1)), MSE)
    # plt.xticks(list(range(0, 41, 2)))
    # plt.xlabel("Number of components")
    # plt.ylabel("MSE")
    # plt.title("Score with varying number of components")
    # plt.show()

    return score, MSE

    # So for this part, I actually wanted to see that the pls model performed dimension reduction
    # but is seems like there's still just as many dimensions. I know that the column headings that I
    # added don't actually correspond to the data shown but we should at least see more 0's.

    # Coefficients currently don't work in PCA.
    # coefficients = pd.DataFrame(coefficients, columns=X_train.columns)
    # print(coefficients)


def main():
    data = pd.read_csv('owid-covid-data.csv')
    shift_var = -1

    uk_data = data.drop(data[data['iso_code'] != "GBR"].index)
    uk_data = uk_data.fillna(0)
    uk_data = uk_data.drop("iso_code", axis = 1)
    uk_data = uk_data.drop("continent", axis = 1)
    uk_data = uk_data.drop("location", axis = 1)

    averages = uk_data.mean()
    num_cases = uk_data[['new_cases']].copy()
    uk_data["new_cases"] = uk_data["new_cases"].shift(shift_var)

    #uk_data["new_cases"] = uk_data["new_cases"].fillna()

    uk_data['date'] = pd.to_datetime(uk_data['date'] )

    uk_data = uk_data.drop("tests_units", axis=1)

#Need to drop columns related to new cases.
    # uk_data['new_cases_smoothed'] = uk_data['new_cases_smoothed'].shift(shift_var)
    # uk_data['new_cases_per_million'] = uk_data["new_cases_per_million"].shift(shift_var)
    # uk_data['new_cases_smoothed_per_million'] = uk_data["new_cases_smoothed_per_million"].shift(shift_var)
    # uk_data['tests_per_case'] = uk_data["tests_per_case"].shift(shift_var)

#new dropped.
    uk_data['date'] = uk_data['date'].map(dt.datetime.toordinal)
    date_var = uk_data['date'].copy()
    uk_data = uk_data.shift(periods = shift_var)

    uk_data['date'] = date_var

    uk_data = uk_data.set_index(uk_data['date'])

    uk_data = uk_data.fillna(averages)
    #uk_data = uk_data.fillna(0)

    pls_score, pls_RMSE = pls_regression(uk_data, num_cases)
    pcr_score, pcr_RMSE = pca(uk_data, num_cases)

    plt.plot(list(range(1, 40, 1)), pls_score, label = "PLS Score")
    plt.plot(list(range(1, 40, 1)), pcr_score, label = "PCR Score")

    # plt.xticks(list(range(0, 41, 2)))
    plt.xlabel("Number of components")
    plt.ylabel("Score (R^2)")
    plt.title("R-squared when predicting 30 days ahead")
    plt.legend()
    plt.show()


    plt.plot(list(range(1, 40, 1)), pls_RMSE, label = "PLS Score")
    plt.plot(list(range(1, 40, 1)), pcr_RMSE, label = "PCR Score")

    # plt.xticks(list(range(0, 41, 2)))
    plt.xlabel("Number of components")
    plt.ylabel("Root mean square error")
    plt.title("RMSE when predicting 30 days ahead")
    plt.legend()
    plt.show()

    print(uk_data.columns)
    print(uk_data)

if __name__=="__main__":
    main()