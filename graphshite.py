import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
ax = plt.gca()

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.title('Graphs of linear functions.')
plt.plot(x, 2*x+1, '-b', label='y=2x+1')
plt.plot(x, 2*x+2, '-g', label='y=2x+2')
plt.plot(x, 2*x+3, '-r', label='y=2x+3')
plt.xlabel('x', labelpad=20)
plt.ylabel('y', labelpad=20)

plt.savefig('graphs.pdf')

plt.show()
