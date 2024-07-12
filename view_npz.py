import numpy as np
import os, sys
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt

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
inputs = np.expand_dims(np.stack(inputs, axis=0), axis=-1)
labels = np.expand_dims(np.stack(labels, axis=0), axis=-1)
names = np.expand_dims(np.stack(names, axis=0), axis=-1)
print(inputs.shape, inputs.dtype)
d={}
d['inputs']= inputs
d['labels']= labels
d['template_names']=names
fn_out = "trainset_"+'x'.join(list(map(str, inputs.shape)))+'.npz'
np.savez(fn_out, **d)

