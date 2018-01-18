import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

data = np.loadtxt('./raw_data_file1.data')
toMatrix = np.mat(data)
# bandwidth
x = toMatrix[:, 2]
# delay
y = toMatrix[:, 3]
# loss
z = toMatrix[:, 4]
# flag
flag = toMatrix[:, 5]
ax.scatter(x[:50], y[:50], z[:50], c='r', marker='x')
ax.scatter(x[500:3500], y[500:3500], z[500:3500], c='b', marker='o')
ax.set_xlabel('Bandwidth(GHz)')
ax.set_ylabel('Delay(ms)')
ax.set_zlabel('Loss(%)')
pl.show()

# for i in range(x_data.shape(0)):
#     for j in range(x_data.shape(1)):
#         if i<50: