
class Derivatives:

    def __init__(self):
        pass

    @staticmethod
    def init_list(end_value):
        i_list = []
        for a in range(0, end_value + 1):
            i_list.append(a)

        return i_list

    @staticmethod
    def derivative_list(initial_list):
        d_list = []
        for value in initial_list:
            d_list.append(value * 3 + 5)

        return d_list


d = Derivatives()

ini = d.init_list(100)

res = d.derivative_list(ini)
print("========= Practice 1: =============")
print(res)
