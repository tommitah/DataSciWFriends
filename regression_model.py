import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from typing import List


def load_file(file_name: str) -> (List[float], List[float]):
    rawdata = np.loadtxt(file_name, delimiter=',', dtype=str)
    data = np.delete(rawdata, 0, 0)
    heights = np.array([], dtype=float)
    weights = np.array([], dtype=float)
    for arr in data:
        heights = np.append(heights, [float(arr[1])], axis=0)
        weights = np.append(weights, [float(arr[2])], axis=0)

    return (heights, weights)


def show_fit(weights, dep_pred):
    r2 = r2_score(weights, dep_pred)
    mse = mean_squared_error(weights, dep_pred)
    mae = mean_absolute_error(weights, dep_pred)
    print('R2: ', r2)
    print('MSE: ', mse)
    print('RMSE: ', np.sqrt(mse))
    print('MAE: ', mae)


def plot_scatter(heights: List[float], weights: List[float]) -> None:

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    height_2D = heights.reshape(-1, 1)
    reg = LinearRegression().fit(height_2D, weights)
    linear = np.linspace(height_2D[0], height_2D[len(
        height_2D)-1], len(heights)).reshape(-1, 1)
    linear_pred = reg.predict(linear)
    dep_pred = reg.predict(height_2D)

    show_fit(weights, dep_pred)

    height_train, height_test, weight_train, weight_test = train_test_split(
        height_2D, weights, test_size=0.2)

    # height is independent and therefore on the x-axis
    plt.scatter(heights, weights, marker='+')
    plt.plot(linear, linear_pred, color='red')
    plt.title('Heights and weights')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.savefig('height_weight_regression.pdf')
    plt.show()
    plt.scatter(height_train, weight_train, marker='+', color='green')
    plt.scatter(height_test, weight_test, marker='+', color='red')
    plt.legend(['train', 'test'])
    plt.title('Dataset splitting')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.savefig('height_weight_regression_dataset_split.pdf')
    plt.show()


def main():
    heights, weights = load_file('weight-height.csv')
    plot_scatter(heights, weights)


if __name__ == "__main__":
    main()
