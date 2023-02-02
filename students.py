import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt


# The moron that I am,
# I completely forgot the existence of numpy for most of this...

# Dataclass module goes nicely with the typing module,
# you can specify struct or class property types as custom structs or
# classes with it, kind of like in TS!
@dataclass
class StudentMeasures:
    '''StudentMeasures(weight: float, height: float)
    dataclass to hold values by student'''
    weight: float
    height: float = 0.0


def check_default(file_name: str):
    try:
        file = open(file_name, 'r')
        file.close()
    except IOError:
        file = open(file_name, 'w')
        file.close()
        print('No file named {} found, created new one.'.format(file_name))
    return file_name


def load_file(file_name):
    rawdata = np.loadtxt(file_name, delimiter=',', dtype=str)
    data = np.delete(rawdata, 0, 0)
    measures = []
    for arr in data:
        measures.append(StudentMeasures(float(arr[2]), float(arr[1])))

    return measures


def convert_measures(data: List[StudentMeasures]):
    for student in data:
        student.weight *= 0.45359237
        student.height *= 2.54


def calculate_means(data: List[StudentMeasures]) -> StudentMeasures:
    weight_sum = 0
    height_sum = 0
    for measure in data:
        weight_sum += measure.weight
        height_sum += measure.height
    return StudentMeasures(weight_sum / len(data), height_sum / len(data))


def calculate_medians(data: List[StudentMeasures]) -> StudentMeasures:
    weights = []
    heights = []
    for measure in data:
        weights.append(measure.weight)
        heights.append(measure.height)
    weights.sort()
    heights.sort()

    median_weight = np.median(weights)
    median_height = np.median(heights)

    return StudentMeasures(median_weight, median_height)


def calculate_std_deviation(data: List[StudentMeasures]) -> StudentMeasures:
    weights = []
    heights = []
    for measure in data:
        weights.append(measure.weight)
        heights.append(measure.height)
    std_weights = np.std(weights)
    std_heights = np.std(heights)

    return StudentMeasures(std_weights, std_heights)


def calculate_variance(data: List[StudentMeasures]) -> StudentMeasures:
    weights = []
    heights = []
    for value in data:
        weights.append(value.weight)
        heights.append(value.height)

    weight_variance = np.var(weights, dtype=np.float32)
    height_variance = np.var(heights, dtype=np.float32)

    return StudentMeasures(weight_variance, height_variance)


def show_histogram(values: List[StudentMeasures], obs: List[float]) -> None:
    plt.style.use('fivethirtyeight')
    plt.title('Student height data')
    plt.hist(values, bins=obs, edgecolor='red')
    plt.savefig('histogram.pdf')
    plt.show()


def main():
    file_name = check_default('weight-height.csv')

    # populate_file(file_name)
    data = load_file(file_name)
    for student in data:
        print('Weight: {:10.2f}lbs, Height: {:10.2f}in'.format(
            student.weight, student.height))
    convert_measures(data)
    means = calculate_means(data)
    print('There were {} students in the data set.'.format(len(data)))
    print('Mean weight: {:10.2f}kg'.format(means.weight))
    print('Mean height: {:10.2f}cm'.format(means.height))
    medians = calculate_medians(data)
    print('Median weight: {:10.2f}kg'.format(medians.weight))
    print('Median height: {:10.2f}cm'.format(medians.height))
    std_deviation = calculate_std_deviation(data)
    print('Std Dev weight: {:10.2f}kg'.format(std_deviation.weight))
    print('Std Dev height: {:10.2f}cm'.format(std_deviation.height))
    variance = calculate_variance(data)
    print('Variance weight: {:10.2f}kg'.format(variance.weight))
    print('Variance height: {:10.2f}cm'.format(variance.height))

    heights = []
    for measure in data:
        heights.append(measure.height)
    show_histogram(heights, [150, 160, 170, 180, 190, 200])


if __name__ == "__main__":
    main()
