import numpy as np


# it appears that arrays should be used when possible instead
# of np.matrix
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])

# Inverse of A
INV_A = np.linalg.inv(A)
# inv_a = a.I

print('A:\n', A)
print('A^-1:\n', INV_A)
product = np.matmul(A, INV_A.astype(int))
print('AA^-1:\n', product)

product = np.matmul(INV_A.astype(int), A)
print('A^-1A:\n', product)

# TODO: These yield odd results, mostly correct, yet one column contains noise
# I don't get it, wolfram alpha confirms that with these matrices the product
# should be  [1 0 0]
#            [0 1 0]
#            [0 0 1]
# I've confirmed that the inverse matrix is correct in both np.array
# and np.matrix but neither matmul() or dot() seem to yield the correct result


# OK: so the crux of this is that we can't mix the matrices with different
# datatypes: both should be integer
# having them one or both as floats fucks us up
