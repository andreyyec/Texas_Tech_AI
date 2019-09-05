import matplotlib.pyplot as plt
import random


class PlotGenerator:

    def __init__(self):
        pass

    @staticmethod
    def _get_3xplus5_value(value):
        return (3 * value) + 5

    @staticmethod
    def get_list_with_random_numbers_added(initial_list):
        transformed_list = []
        for i in initial_list:
            transformed_list.append(i + random.randrange(-5, 6))
        return transformed_list

    @staticmethod
    def generate_unidimensional_plot(initial_list, label, format_string=""):
        plt.plot(initial_list, format_string)
        plt.ylabel(label)
        plt.show()

    @staticmethod
    def generate_bidimensional_plot(x_list, y_list, label):
        plt.plot(x_list, x_list, "b:", x_list, y_list, "r.")
        plt.ylabel(label)
        plt.show()

    def get_transformed_list(self, initial_list):
        transformed_list = []
        for i in initial_list:
            transformed_list.append(self._get_3xplus5_value(i))
        return transformed_list


pg = PlotGenerator()

ini_list = range(-10, 11)

pg.generate_unidimensional_plot(ini_list, "Plot showing y=3x+5")

trans_list = pg.get_transformed_list(ini_list)

rand_trans_list = pg.get_list_with_random_numbers_added(trans_list)

print("========= Practice 2: A  =============")
print(trans_list)

print("========= Practice 2: B  =============")
pg.generate_unidimensional_plot(rand_trans_list, "Plot showing y=3x+5 plus rand values", "r.")

print("========= Practice 2: C  =============")
pg.generate_bidimensional_plot(trans_list, rand_trans_list, "X/Y scatter plot y=3x+5")

