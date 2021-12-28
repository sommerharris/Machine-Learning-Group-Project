
import math

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


def pls_regression(X, y, number_of_components):


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, train_size=0.8)
    pls = PLSRegression(n_components= number_of_components)
    pls.fit(X_train, y_train)

    pls_score = pls.predict(X_test)
    plt.plot(list(range(0, len(y_test), 1)), pls_score, label = "Prediction")
    plt.plot(list(range(0, len(y_test), 1)), y_test, label = "Actual")
    plt.legend()
    plt.show()




def main():
    data = pd.read_csv('owid-covid-data.csv')
    shift_var = -30

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

    pls_score, pls_RMSE = pls_regression(uk_data, num_cases, 10)






if __name__=="__main__":
    main()