import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

img_size = 21
input_shape = [img_size, img_size, 1]
alpha = 0.9

colors = np.empty(input_shape+[4], dtype=np.float32)
colors[:] = [1,1,1, alpha]
#colors[0] = [1, 0, 0, alpha]  # R
#colors[1] = [0, 1, 0, alpha]  # G
#colors[2] = [0, 0, 1, alpha]  # B
#colors[3] = [1, 1, 0, alpha]  # yellow
#colors[4] = [1, 1, 1, alpha]  # grey
#colors[:] = [1, 1, 1, alpha]

L = range(img_size)
X, Y = np.meshgrid(L, L)
Z = np.sin(0.2*X)**2 * np.cos(0.2*Y)**2

# broadcating
colors[:,:,0,0] = 1-Z
colors[:,:,0,1] = 1-Z
colors[:,:,0,2] = 1-Z

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(131, projection='3d')
ax1.voxels(np.ones(input_shape), facecolors=colors, edgecolors='grey')
ax1.set_box_aspect((1, 1, 0.05))
ax1.set_axis_off()

ax2 = fig.add_subplot(132, projection='3d')
ax2.voxels(np.ones([11,11, 4]), facecolors='white', edgecolors='grey')
ax2.set_box_aspect(( 1, 1, 0.2))
ax2.set_xlim(0,21)
ax2.set_ylim(0,21)
ax2.set_axis_off()

ax4 = fig.add_subplot(133, projection='3d')
ax4.voxels(np.ones([11,11, 1]), facecolors='white', edgecolors='grey')
ax4.set_box_aspect(( 1, 1, 0.05))
ax4.set_xlim(0,21)
ax4.set_ylim(0,21)
ax4.set_axis_off()


plt.show()
