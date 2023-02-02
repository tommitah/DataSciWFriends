import matplotlib.pyplot as plt
import numpy as np


def main():
    table_one = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    table_two = np.array([-0.57, -2.57, -4.80, -7.36, -
                          8.78, -10.52, -12.85, -14.69, -16.78])

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.scatter(table_one, table_two, marker='+')
    plt.savefig('scatterings.pdf')
    plt.show()


if __name__ == "__main__":
    main()
