import numpy as np
import os, sys
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
import tensorflow as tf

files = sys.argv[1:]
names=[]
inputs=[]
labels=[]
for fn in files:
    bn = basename = os.path.basename(fn)
    f, ext = os.path.splitext(bn)
    data = np.load(fn)
    field_in = data['input_layers']
    field_out = data['output_layers']
    names.append(f)
    inputs.append(field_in)
    labels.append(field_out)

inputs = np.stack(inputs, axis=0)
labels = np.stack(labels, axis=0)
names = np.array(names)
print(names.shape)

images = np.stack([inputs, labels], axis=-1)
print(images.shape)

Lx = np.arange(0, 992-256, 200)
X, Y = np.meshgrid(Lx, Lx)
boxes = list(zip(X.flatten(), Y.flatten()))

clips=[]
for (x0, y0) in boxes:
    clip = tf.image.crop_to_bounding_box(images, x0, y0, 256, 256)
    clips.append(clip.numpy())

clips = np.concatenate(clips, axis=0)

res, = np.where(np.sum(clips[:,:,:,0], axis=(1,2)) >0)
clips = clips[res]
print(res)
print(clips.shape, clips.dtype)




print(inputs.shape, inputs.dtype)
d={}
#d['inputs']= clips[:,:,:,0]
#d['labels']= clips[:,:,:,1]
d['images'] = clips
d['template_names']=names
fn_out = "trainset_"+'x'.join(list(map(str, clips.shape)))+'.npz'
np.savez(fn_out, **d)

