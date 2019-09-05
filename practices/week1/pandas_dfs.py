import pandas as pd


class PandasExample:

    def __init__(self):
        df = self.load_csv_file("./static_files/PeriodicTable.csv")
        df1 = self.create_derivative_column('ratio', "atomic mass", "atomic number")
        self.show_data_frame(df1)

    @staticmethod
    def create_derivative_column(data_frame, column_name, column1, column2):
        data_frame[column_name] = column1 / column2
        return data_frame

    @staticmethod
    def show_data_frame(csv_file):
        print(csv_file)

    @staticmethod
    def load_csv_file(path):
        return pd.read_csv(path)