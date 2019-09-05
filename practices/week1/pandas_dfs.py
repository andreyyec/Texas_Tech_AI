import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class PandasExample:

    def __init__(self):
        self._df1 = pd.read_csv("./datasets/PeriodicTable.csv")
        self._df2 = pd.read_csv("./datasets/craigslistVehicles.csv")

    def get_periodic_table_example(self):
        df = self.create_derivative_column(self._df1, 'ratio', "atomic mass", "atomic number")
        print(self._df1.info())
        print(df)

    def get_craiglist_vehicles_numeric_columns_datagrams(self):
        numeric_df = self._df2.select_dtypes(include=np.number)
        columns = list(numeric_df)

        print(self._df2)

        for i in columns:
            print(i)
            numeric_df.hist(i, bins=50)
            plt.show()

    def generate_x_y_plots(self):
        plt.scatter(self._df2["year"], self._df2["price"])
        plt.show()

        plt.scatter(self._df2["odometer"], self._df2["price"])
        plt.show()

    @staticmethod
    def create_derivative_column(df, col_name, col1, col2):
        df[col_name] = df[col1] / df[col2]
        return df


pds = PandasExample()

pds.get_periodic_table_example()

pds.get_craiglist_vehicles_numeric_columns_datagrams()

pds.generate_x_y_plots()
