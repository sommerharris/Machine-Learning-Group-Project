import pandas as pd
import numpy as np

def main():


    data = pd.read_csv('owid-covid-data.csv')
    print(data.columns)


    data_top = data.head()

    nan_columns = data.columns[data.isnull().any()]
    print(nan_columns)

    # np.where can be used when there is only a single condition
    #data['continent'] = np.where(data['continent'].isnull()  , data['location'], data['continent'])

    #Total cases is mostly likely 0 if it is NULL
    data['total_cases'] = np.where(data['total_cases'].isnull(), data['total_cases'], 0)
    data['new_cases'] = np.where(data['new_cases'].isnull(), data['new_cases'], 0)

    #How to remove rows from the dataset.
    data = data.drop(data[data['location'] == "Africa"].index)
    data = data.drop(data[data['location'] == "Europe"].index)


    print(data.continent.unique())
    print(data.location.unique())

main()





