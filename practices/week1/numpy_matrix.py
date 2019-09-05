import numpy as np


class NumpyMatrix:

    def __init__(self):
        pass

    @staticmethod
    def get3by3matrix():
        return np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])

    @staticmethod
    def get_inverse_matrix(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def multiply_matrix(matrix1, matrix2):
        return matrix1 @ matrix2


npM = NumpyMatrix()

ini_matrix = npM.get3by3matrix()

inv_matrix = npM.get_inverse_matrix(ini_matrix)

id_matrix = npM.multiply_matrix(ini_matrix, inv_matrix)

print("========= Practice 3: A =============")
print(ini_matrix)

print("========= Practice 3: B =============")
print(inv_matrix)

print("========= Practice 3: C =============")
print(id_matrix)

