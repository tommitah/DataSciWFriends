import numpy as np


a = np.array([[1, 2], [3, 4]])
b = np.array([[-1, 1], [5, 7]])

print('Determinant of A: ', np.linalg.det(a))
print('Determinant of B: ', np.linalg.det(b))
print('Determinant of product AB: ', np.linalg.det(np.matmul(a, b)))
