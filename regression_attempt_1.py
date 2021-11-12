import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

# https://stackoverflow.com/questions/34007308/linear-regression-analysis-with-string-categorical-features-variables

#date data isn't good for the standard linear regression package
import datetime as dt

'''
Attempt to regress on UK covid data.
'''
def main():
    data = pd.read_csv('owid-covid-data.csv')

    data = data.drop(data[data['iso_code']!="GBR"].index)
    print(data.location.unique())


    x = data[['new_cases', 'date', 'reproduction_rate', 'new_tests', 'stringency_index']].copy()

    x['reproduction_rate'] = x['reproduction_rate'].fillna(0)
    x['new_tests'] = x['new_tests'].fillna(0)
    x['stringency_index'] = x['stringency_index'].fillna(0)

    x['date'] = pd.to_datetime(x['date'])
    x['date'] = x['date'].map(dt.datetime.toordinal)

    y = data[['new_cases']].copy()
    x = x.drop('new_cases', axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lin_model = LinearRegression(n_jobs=4)
    lin_model.fit(x_train, y_train)

    y_predict = lin_model.predict(x_test)

    r2 = r2_score(y_pred=y_predict, y_true=y_test)

    print(x.head())
    print(y.head())
    nan_columns = x.columns[x.isnull().any()]

    print(nan_columns)

    print("R2 score OLS regression")
    print(r2)

    l2_lin_model = Ridge(fit_intercept=0)
    l2_lin_model.fit(x_train, y_train)
    l2_predict = l2_lin_model.predict(x_test)

    l2_r2 = r2_score(y_pred=l2_predict, y_true=y_test)
    print("Ridge score regression")
    print(l2_r2)


main()
