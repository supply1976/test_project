import os, sys
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
from PIL import Image


class TileBlock:
  def __init__(self, block_size=(32, 32)):
    # unit in pixel
    self.block_h, self.block_w = block_size
    self.zero_block = np.zeros(shape=block_size, dtype=np.uint8)

  def gen_rectangle(self, min_rec_size=(8, 8), random_empty=False):
    """
    min_rec_size: unit in pixel
    generate random rectangle within tile
    by random integers for ll and ur coordinates
    return: tile (2D int array with shape=tile_size)
            with value 1 (inside rectangle) or 0 (outside rectangle)
    """
    block = None
    if min_rec_size is None:
      return self.zero_block
    h, w = min_rec_size
    assert h in range(1, self.block_h+1)
    assert w in range(1, self.block_w+1) 
    x0 = np.random.randint(0, self.block_w - (w-1))
    x1 = np.random.randint(x0 + (w-1), self.block_w)
    y0 = np.random.randint(0, self.block_h - (h-1))
    y1 = np.random.randint(y0 + (h-1), self.block_h)
    # rectangle coordinates 
    ll = (x0, y0)
    ur = (x1, y1)
    #print("ll={}, ur={}".format(ll, ur))
    # rectangle array
    rec = np.ones(shape=(y1-y0+1, x1-x0+1), dtype=np.uint8)
    # zero padding to rectangle array 
    block = np.pad(rec, ((y0, self.block_h-y1-1), (x0, self.block_w-x1-1)))
    assert block.shape == (self.block_h, self.block_w)
    if random_empty:
      rid = np.random.randint(2)
      block = [block, self.zero_block][rid]
    return block


def create_random_pattern(block_size, num_blocks, min_rec_size, random_empty=False):
  tb = TileBlock(block_size)
  blocks=[]
  for i in range(num_blocks):
    _block=[]
    for j in range(num_blocks):
      rec1 = tb.gen_rectangle(min_rec_size, random_empty)
      rec2 = tb.gen_rectangle(min_rec_size, random_empty)
      block = np.logical_or(rec1, rec2).astype(np.uint8)
      _block.append(block)
    blocks.append(_block)
  p = np.block(blocks)
  return p


# example
fig, axes = plt.subplots(nrows=2, ncols=2)
axes = axes.flatten()

p1 = create_random_pattern(block_size=(32,32), min_rec_size=(8,4), num_blocks=4)
p2 = create_random_pattern(block_size=(32,32), min_rec_size=(8,4), num_blocks=4)
p3 = create_random_pattern(block_size=(32,32), min_rec_size=(4,8), num_blocks=4)
p4 = create_random_pattern(block_size=(32,32), min_rec_size=(4,8), num_blocks=4)

p=[p1, p2, p3, p4]

for i in range(4):
  #axes[i].pcolormesh(p[i])
  axes[i].imshow(p[i], interpolation='none')
  axes[i].set_aspect('equal')

p = np.block([[p1, p2], [p3, p4]])
#im = Image.fromarray(p*255, 'L')
#im.save('random_pattern.jpg')
plt.tight_layout()
plt.show()


# create 100 pattern, save to numpy array
images=[]
img_dir = 'test_patterns'
if not os.path.isdir(img_dir):
  os.mkdir(img_dir)

for i in range(100):
  img_name = os.path.join(img_dir, 'test_'+str(i+1).zfill(4)+'.jpg')
  p = create_random_pattern(block_size=(32,32), num_blocks=4, min_rec_size=(4,4))
  im = Image.fromarray(p*255, 'L')
  im.save(img_name)
  images.append(p)

images = np.stack(images, axis=0)
print(images.shape)
np.save('random_patterns.npy', images)
