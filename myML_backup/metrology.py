import sys, os
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d, RectBivariateSpline
import scipy.optimize as scpy_opt
#from collections import OrderedDict


def _getGaugeInfo(asd_file):
  A = pd.read_csv(asd_file, skipinitialspace=True, sep="\s+", comment="'")
  HG_idx, = np.where(A['base_y']==A['head_y'])
  VG_idx, = np.where(A['base_x']==A['head_x'])
  return (HG_idx, VG_idx, A)
  

def _getTslices(images, HG_idx, VG_idx):
  if len(images.shape)==4:
    images = np.squeeze(images)
  nums, h, w = images.shape
  print("images {}".format(images.shape))
  print("# of HG {}, # of VG {}".format(len(HG_idx), len(VG_idx)))
  Tslices = None
  if len(HG_idx)==0:
    Tslices = images[:, :, w//2]
  elif len(VG_idx)==0:
    Tslcies = images[:, h//2, :]
  else:
    TslicesH = images[HG_idx, h//2, :]
    TslicesV = images[VG_idx, :, w//2]
    Tslices = np.concatenate([TslicesH, TslicesV], axis=0)
    print("Tslices {}".format(Tslices.shape))
    idx = np.argsort(np.concatenate([HG_idx, VG_idx]))
    Tslices = Tslices[idx]
  return Tslices


def _extractCDFromCenter(tslices, threshold, grain_size):
  """Extract model CDs from the tslices
     Parameters
     ----------
     tslices : 2D numpy array (signal),
     threshold : model threshold

     Returns:
     --------
     simCD : 1D numpy array
     unresolved : list of indices where tslice has no cut with threshold.
  """
  
  simCD, unresolved = [], []
  for t_i, tslice in enumerate(tslices) :
    n = len(tslice)
    # shift the array so that the middle element represents coordinate x=0
    x = np.arange(n) - n//2
    f = interp1d (x, tslice-threshold, kind='cubic')

    xL, xR = np.nan, np.nan
    # find the first root at the left-hand side of the center
    for i in range(n//2) :
      if f(0) * f(-i) <=0 :
        xL = scpy_opt.brentq (f, -i, 0)
        break

    # find the first root at the right-hand side of the center
    for i in range(n//2) :
      if f(0) * f(i) <= 0 :
        xR = scpy_opt.brentq (f, 0, i)
        break

    cd = xR - xL if not np.isnan(xR) and not np.isnan(xL) else np.nan
    if np.isnan (cd) :
      unresolved.append(t_i)

    simCD.append(grain_size*cd)
  return np.array(simCD), unresolved


def load_data(npz):
  db = np.load(npz)
  train_inputs = db['train_inputs']
  train_targets = db['train_targets']
  valid_inputs = db['valid_inputs']
  valid_targets = db['valid_targets']
  return (train_inputs, train_targets, valid_inputs, valid_targets)


def main():
  npz_file = './database.npz'
  asd_file = './g2712.asd'
  threshold = 0.3988341
  grain_size = 8.0
  # input: optical 
  # target: resist
  train_inputs, train_targets, valid_inputs, valid_targets = load_data(npz_file)

  hgs, vgs, A = _getGaugeInfo(asd_file)

  resistTslices = _getTslices(valid_targets, hgs, vgs)
  opticalTslices = _getTslices(valid_inputs, hgs, vgs)
  opt_thresh = np.mean(opticalTslices)

  print("optical threshold = {}".format(opt_thresh))
  print("resist threshold = {}".format(threshold))

  resistSimCD, unresolved_RI = _extractCDFromCenter(resistTslices, threshold, grain_size)
  opticalSimCD, unresolved_AI = _extractCDFromCenter(opticalTslices, opt_thresh, grain_size)
  print("{} unresolved gauges in resist signal".format(len(unresolved_RI)))
  print("{} unresolved gauges in optical signal".format(len(unresolved_AI)))

  A['resistSimCD']=resistSimCD
  A['opticalSimCD']=opticalSimCD

  A['fiterr_RI']=A['resistSimCD'] - A['wafer_CD']
  A['fiterr_AI']=A['opticalSimCD'] - A['wafer_CD']

  Ag = A.groupby('Myclass')
  RImodelQoR = Ag['fiterr_RI'].agg(["std", "max", "min", "mean", "count"])
  AImodelQoR = Ag['fiterr_AI'].agg(["std", "max", "min", "mean", "count"])
  print(RImodelQoR)
  print(AImodelQoR)

if __name__=="__main__":
  main()

  
