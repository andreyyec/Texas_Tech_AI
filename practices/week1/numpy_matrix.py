import numpy as np


class NumpyMatrix:

    def __init__(self):
        pass

    @staticmethod
    def multiply_matrix(matrix1, matrix2):
        return np.dot(matrix1, matrix2)

    @staticmethod
    def get_reverse_matrix(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def get3by3matrix(self):
        return np.array([1, 2, 3], [3, 1, 2], [2, 3, 1])


