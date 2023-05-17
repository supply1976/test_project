import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    block_h, block_w = (100, 100)
    nx, ny = (10, 10)
    max_tiles = nx * ny

    arr = np.ones(shape=(ny, block_h, nx, block_w), dtype=np.uint8)
    b = np.random.choice([0, 255], size=max_tiles)
    print(b)
    b = b.reshape([ny, 1, nx, 1])
    arr = (arr * b).astype(np.uint8)
    arr_1 = arr.reshape(ny*block_h, nx*block_w)
    print(arr_1.dtype)

    plt.imshow(arr_1)
    plt.show()

main()

    

