from proteus import simulation as psim
from proteus import fops
from proteus.fops import utility as fopsutil
from proteus import ilt
from proteus import layer as proteus_layer
from proteus import layout as proteus_layout
from proteus.ilt import levelset
from proteus._cruise.simlib.v1 import _gauss
import argparse
import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt


def get_field_image(fopslayer, pitch=2.0):
    bbox = fopslayer.value.boundingBox()
    grid = fopsutil.fieldForBoundingBox(bbox, 
        pitch=pitch, square=True, power_of_two=True)
    shape = grid.shape
    pitch = grid.pitch
    w, s, e, n = bbox
    cx, cy = (w+e)/2.0, (s+n)/2.0
    origin=(cx- shape[0]*pitch//2, cy-shape[0]*pitch//2)
    print(shape)
    print(bbox, cx, cy)
    # rasterized field
    fld1 = fops.rasterize(fopslayer, grid, raster_kernel=fops.ETCH_KERNEL, 
        algorithm=fops.RasterizationAlgorithm.POLYLINE_RASTERIZE)
    arr = fld1.value
    return arr


def gen_clips(df, clip_size, lh, ldt, get_vertices=False):
    if len(df)>10000:
        print("reading more than 10,000 gauges...")
    
    for i, ld in enumerate(ldt):
        ld = ":".join(list(map(str, ld)))
        print('read input layer ID {}: "{}"'.format(i+1, ld))
    
    vertex_info = {}
    for k in ldt: 
        vertex_info[k] = []
    
    clips = []
    images = []
    for i, (x0, y0) in enumerate(df[['base_x', 'base_y']].values):
        if i%500==0:
            print("processing... {}/{}".format(i, len(df)))
        llx, lly = (x0-clip_size//2, y0-clip_size//2)
        urx, ury = (x0+clip_size//2, y0+clip_size//2)
        clip_box = (llx, lly, urx, ury)
        clip = proteus_layer.readOasGds(
            lh, 
            layer_dt = ldt,
            offset = None,
            clip_box = clip_box,
            shape_type = 'auto',
            new_api = True)
        clips.append(clip)
        print(i, (x0, y0))
        fopslayer = fops.Layer(clip[ldt[0]])
        arr = get_field_image(fopslayer)
        images.append(arr)

        if get_vertices:    
            for ld in ldt:
                layer_i = clip[ld]
                nVertex = layer_i.numVertices()
                vertex_info[ld].append(nVertex)
    images = np.stack(images, axis=0)
    return (images, vertex_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, required=True, help="layout file")
    parser.add_argument('--asd', type=str, required=True, help=".asd file")
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--clip_size', type=int, default=100)
    parser.add_argument('--save_fields', dest='save_fields', action='store_true')

    FLAGS, _ = parser.parse_known_args()
    clip_size = FLAGS.clip_size
    
    # files check
    if not os.path.isfile(FLAGS.asd):
        print("{} not found".format(FLAGS.asd))
        return 

    if not os.path.isfile(FLAGS.layout):
        print("{} not found".format(FLAGS.layout))
        return 
    
    _, ext = os.path.splitext(FLAGS.layout)
    if ext !='.oas' and ext !='.gds':
        print("not support")
        return 

    # layer data check
    ldt = FLAGS.layers
    if len(ldt)%2 !=0:
        print("layers should be a list of even number integers")
        return 
    ldt = list(zip(ldt[::2], ldt[1::2]))
    
    # read asd file as a dataframe
    asd_file = os.path.abspath(FLAGS.asd)
    dirname = os.path.dirname(asd_file)
    fn, ext = os.path.splitext(os.path.basename(asd_file))
    
    A = pd.read_csv(asd_file, sep='\s+', comment="'")
    print("\nread asd file: {} with {} gauges".format(fn+ext, len(A)))
    
    # read layout file as LayoutHolder object
    layoutHolder = proteus_layout.LayoutHolder(FLAGS.layout)
  
    print("generating clips, clip size = {}".format(clip_size))
    images, vertex_info = gen_clips(A, clip_size, layoutHolder, ldt, get_vertices=True)
    print(images.shape)

    for k in vertex_info:
        A['clip_size'] = clip_size
        ks = "nVertex_"+":".join(list(map(str, k)))
        A[ks] = vertex_info[k]

    # save to new asd file with layer info
    output_asd = os.path.join(dirname, fn+"_updated"+ext)
    A.to_csv(output_asd, sep='\t', na_rep="NA", index=False)
    print("Done")
    print("update vertex info to {}".format(output_asd))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i], origin='lower')


main()
plt.show()

        


