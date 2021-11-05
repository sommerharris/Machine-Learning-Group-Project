import pandas as pd


def main():


    data = pd.read_csv('owid-covid-data.csv')
    print(data.columns)


    data_top = data.head()

    nan_columns = data.columns[data.isnull().any()]
    print(nan_columns)


    print("Hello World!\n")
main()





