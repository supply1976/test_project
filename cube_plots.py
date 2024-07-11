import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

input_arr = np.load("mario_pixel_32x32.npy")
label_arr = np.load("mario2_pixel_11x11.npy")

img_size = 32
input_shape = [img_size, img_size, 1]
alpha = 0.9

colors1 = np.empty(input_shape+[4], dtype=np.float32)
colors1[:] = [1,1,1, alpha]
#colors[0] = [1, 0, 0, alpha]  # R
#colors[1] = [0, 1, 0, alpha]  # G
#colors[2] = [0, 0, 1, alpha]  # B
#colors[3] = [1, 1, 0, alpha]  # yellow
#colors[4] = [1, 1, 1, alpha]  # grey
#colors[:] = [1, 1, 1, alpha]

L = range(img_size)
#X, Y = np.meshgrid(L, L)
#Z = np.sin(0.2*X)**2 * np.cos(0.2*Y)**2
Z1 = np.rot90(input_arr, k=-1)
# broadcating
colors1[:,:,0,0] = Z1
colors1[:,:,0,1] = Z1
colors1[:,:,0,2] = Z1

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(141, projection='3d')
ax1.voxels(np.ones(input_shape), facecolors=colors1, edgecolors='grey')
ax1.set_box_aspect((1, 1, 0.04))
ax1.set_axis_off()
ax1.set_title("input")


ax2 = fig.add_subplot(142, projection='3d')
ax2.voxels(np.ones([24,24, 4]), facecolors='white', edgecolors='grey')
ax2.set_box_aspect(( 1, 1,0.16))
ax2.set_xlim(0,32)
ax2.set_ylim(0,32)
ax2.set_axis_off()
ax2.set_title("nth layer output")

ax3 = fig.add_subplot(143, projection='3d')
ax3.voxels(np.ones([18,18, 8]), facecolors='white', edgecolors='grey')
ax3.set_box_aspect(( 1, 1, 0.32))
ax3.set_xlim(0,32)
ax3.set_ylim(0,32)
ax3.set_axis_off()
ax3.set_title("(n+1)th layer output")


Z2 = np.rot90(label_arr, k=-1)
colors2 = np.empty([11,11,1,4], dtype=np.float32)
colors2[:] = [1,1,1, alpha]
# broadcating
colors2[:,:,0,0] = Z2
colors2[:,:,0,1] = Z2
colors2[:,:,0,2] = Z2
ax4 = fig.add_subplot(144, projection='3d')
ax4.voxels(np.ones([11,11, 1]), facecolors=colors2, edgecolors='grey')
ax4.set_box_aspect((1, 1, 0.04))
ax4.set_xlim(0,32)
ax4.set_ylim(0,32)
ax4.set_axis_off()
ax4.set_title("final output")

plt.tight_layout()
plt.show()
