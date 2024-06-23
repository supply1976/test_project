import os, sys
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
from PIL import Image
from functools import reduce


class BlockUnit:
    def __init__(self, block_size=(32, 32)):
        # unit in pixel
        self.block_size = block_size
        self.zero_block = np.zeros(shape=block_size, dtype=np.uint8)


    def gen_random_rec(self, min_rec=(8, 8), max_rec=None):
        """
        min_rec: unit in pixel
        generate rectangle with random size between min_rec to block_size
        return: 
        rec: 2D int array with shape = block_size 
        with value 1 (inside rectangle) or 0 (outside rectangle)
        """
        rec = None
        min_h, min_w = min_rec
        if max_rec is None:
            max_h, max_w = self.block_size
        else:
            max_h, max_w = max_rec
        assert min_h in range(1, self.block_size[0]+1)
        assert min_w in range(1, self.block_size[1]+1)
        assert max_h > min_h
        assert max_w > min_w
        h = np.random.randint(min_h, max_h)
        w = np.random.randint(min_w, max_w)
        rec = np.ones(shape=(h, w), dtype=np.uint8)
        # random pad zero to rec upto size = block_size
        pad_n = np.random.randint(0, self.block_size[0]-h+1)
        pad_s = self.block_size[0] - h - pad_n
        pad_w = np.random.randint(0, self.block_size[1]-w+1)
        pad_e = self.block_size[1] - w - pad_w
        pad_width = [[pad_n, pad_s], [pad_w, pad_e]]
        #print(rec.shape, pad_width)
        rec = np.pad(rec, pad_width=pad_width)
        assert rec.shape == self.block_size
        return rec

    def gen_random_block(self, min_rec, max_rec, nums_rec):
        recs = None
        block = None
        if nums_rec > 1:
            recs = [self.gen_random_rec(min_rec, max_rec) for _ in range(nums_rec)]
            block = reduce(lambda a,b: np.logical_or(a,b).astype(np.uint8), recs)
        else:
            block = self.gen_random_rec(min_rec)
        return (recs, block)


def create_random_pattern(
    block_size, num_blocks, min_rec_size, periodic=False, random_empty=False):
    
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
    if periodic:
        p = np.tile(blocks[0][0], (num_blocks, num_blocks))
    return p


def example1():
    fig, axes = plt.subplots(nrows=2, ncols=4)
    axes = axes.flatten()
    p1 = create_random_pattern(block_size=(32,32), min_rec_size=(4,4), num_blocks=2)
    p2 = create_random_pattern(block_size=(32,32), min_rec_size=(4,4), num_blocks=2)
    p3 = create_random_pattern(block_size=(32,32), min_rec_size=(4,4), num_blocks=2)
    p4 = create_random_pattern(block_size=(32,32), min_rec_size=(4,4), num_blocks=2)

    p = [p1, p2, p3, p4]

    for i in range(len(p)):
        print(p[i].shape)
        #axes[i].pcolormesh(p[i])
        axes[i].imshow(p[i], interpolation='none')
        axes[i].set_aspect('equal')


def example2(save_dir, nums=100, save_to_jpg=True):
    # create 100 patterns, save to numpy array
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    
    images=[]
    for i in range(nums):
        img_name = os.path.join(save_dir, 'random_polygon_'+str(i+1).zfill(4)+'.jpg')
        p = create_random_pattern(
            block_size=(32,32), num_blocks=4, min_rec_size=(4,4), periodic=True)
        im = Image.fromarray(p*255, 'L')
        if save_to_jpg:
            im.save(img_name)
        images.append(p)

    images = np.stack(images, axis=0)
    images = np.expand_dims(images, axis=-1)
    print(images.shape)
    fn = "random_patterns_"+"x".join(map(str, images.shape))+".npy"
    np.save(fn, images)
    return images
  

if __name__=="__main__":
    MIN_REC = (20, 5)
    MAX_REC = (40,15)
    BLOCK_SIZE = (64, 64)

    bu = BlockUnit(BLOCK_SIZE)
    #rec = bu.gen_random_rec(min_rec=MIN_REC)

    blocks = []
    for i in range(4):
        _blocks=[]
        for j in range(4):
            _, block = bu.gen_random_block(min_rec=MIN_REC, max_rec=MAX_REC,
            nums_rec=4)
            _blocks.append(block)
        blocks.append(_blocks)

    p = np.block(blocks)
    print(p.shape)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
    axes[0].imshow(blocks[0][0])
    axes[1].imshow(blocks[0][1])
    axes[2].imshow(p)
    plt.show()
    np.save("test_pattern.npy", p) 
    #example1()
    #example2('random_patterns_2', nums=100)
    #plt.show()


