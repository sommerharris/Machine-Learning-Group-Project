
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib as plt

#https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
def pcr_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for i in range(1, 40, 2):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")
        print(pls.score(X_test, y_test))

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