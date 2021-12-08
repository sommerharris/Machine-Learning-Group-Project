


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

def pls_regression(X_train, X_test, y_train, y_test):

    for i in range(1, 40, 3):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")
        y_pred = pls.predict(X_test)

        print(f"MSE: {mean_squared_error(y_true = y_test, y_pred = y_pred, squared=False)}")


def main():
    shift_var = -1
    data = pd.read_csv('owid-covid-data.csv')
    global_data = data[data['iso_code'].notna()]

    global_data = global_data.fillna(0)



    global_data = global_data.drop("tests_units", axis=1)

    global_data['date'] = pd.to_datetime(global_data['date'])
    global_data['date'] = global_data['date'].map(dt.datetime.toordinal)

    # Test shifting the data.
    global_data['new_cases_smoothed'] = global_data['new_cases_smoothed'].shift(shift_var)


    global_data['new_cases_per_million'] = global_data["new_cases_per_million"].shift(shift_var)

    global_data['new_cases_smoothed_per_million'] = global_data["new_cases_smoothed_per_million"].shift(shift_var)

    global_data = global_data.drop("tests_per_case", axis=1)

    global_data = global_data.fillna(0)


    #Test dropping components
    global_data = global_data.drop("total_cases", axis=1)
    global_data = global_data.drop("total_cases_per_million", axis = 1)


    canada_data = global_data.loc[global_data['iso_code'] == "CAN"]
    can_num_cases = canada_data[['new_cases']].copy()
    canada_data["new_cases"] = canada_data["new_cases"].shift(shift_var)
    canada_data = canada_data.fillna(0)

    global_data = global_data.drop(global_data[global_data['iso_code'] == "CAN"].index)

    #num cases
    num_cases = global_data[['new_cases']].copy()
    global_data["new_cases"] = global_data["new_cases"].shift(shift_var)
    global_data = global_data.fillna(0)


    global_data = global_data.drop("iso_code", axis=1)
    global_data = global_data.drop("continent", axis=1)
    global_data = global_data.drop("location", axis=1)


    canada_data = canada_data.drop("iso_code", axis=1)
    canada_data = canada_data.drop("continent", axis=1)
    canada_data = canada_data.drop("location", axis=1)

    pls_regression(global_data, canada_data, num_cases, can_num_cases)


    print(global_data.columns)


if __name__=="__main__":
    main()