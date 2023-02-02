import numpy as np
from matplotlib import pyplot as plt


def main():
    for n in [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]:
        sums = []
        for i in range(n):
            die = np.random.randint(1, 7, size=(1, 2))
            sums.append(np.sum(die))
        hist, bins = np.histogram(sums, range(2, 14))
        print('Variance v: {} for n: {}'.format(
            np.var(sums, dtype=np.float32), n))
        plt.bar(bins[:-1], hist/n)
        plt.title('Dice throws where n is {}'.format(n))
        plt.show()


if __name__ == "__main__":
    main()
