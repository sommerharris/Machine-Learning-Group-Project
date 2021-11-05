import pandas as pd


def main():


    data = pd.read_csv('owid-covid-data.csv')
    print(data.columns)

    data_top = data.head()


    print("Hello World!\n")
main()





