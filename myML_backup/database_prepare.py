import os, sys
import numpy as np
import pandas as pd


# real deCNN .npz and return images
def get_images(fn):
    data = np.load(fn, allow_pickle=True, encoding='latin1')
    trimages = np.stack(data['trimages'], axis=0)
    print(trimages.shape)
    trimagesE1 = trimages[:,0,:,:]
    trimagesE2 = trimages[:,1,:,:]
    trimages = np.concatenate([trimagesE1, trimagesE2])
    trimages = np.expand_dims(trimages, axis=-1)
    print(trimages.shape, trimages.dtype)
    vlimages = np.concatenate(data['vlimages'], axis=0)
    vlimages = np.expand_dims(vlimages, axis=-1)
    print(vlimages.shape, vlimages.dtype)
    df = data['asd'][()]
    #keys = list(map(lambda x: x.decode(), keys))
    df = pd.DataFrame(df)
    print(df.head())
    dfg = df.groupby("Myclass")
    print(dfg.size())
    return (trimages, vlimages)




fn1 = sys.argv[1]
fn2 = sys.argv[2]

# optical 
train_inputs, valid_inputs = get_images(fn1)

# resist
train_targets, valid_targets = get_images(fn2)

d={}
d['train_inputs']=train_inputs
d['train_targets']=train_targets
d['valid_inputs']=valid_inputs
d['valid_targets']=valid_targets

np.savez('database.npz', **d)


