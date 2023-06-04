import os, sys
import numpy as np
import matplotlib as mpl
#mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


#------------------------------------------------------------------------------------#
# Math Kernels (non-trainable) for (space) gradient operations
#------------------------------------------------------------------------------------#
class MathKernels(object):
  @staticmethod
  def _sobelx(inputTensor4D, padding='VALID'):
    """ Sobel gradx filter 3x3 """
    sobelx = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]
    kgx = tf.reshape(tf.constant(sobelx, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgx, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _gradx_3x3(inputTensor4D, padding='VALID'):
    # central difference 3x3, uniform grid
    gx = [[ 0, 0, 0],
          [-1, 0, 1],
          [ 0, 0, 0]]
    gx = (1/2.0) * np.array(gx)
    kgx = tf.reshape(tf.constant(gx, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgx, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _gradx_5x5(inputTensor4D, padding='VALID'):
    # central difference 5x5, uniform grid
    gx = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1,-8, 0, 8,-1],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]
    gx = (1/12.0) * np.array(gx)
    kgx = tf.reshape(tf.constant(gx, tf.float32), [5,5,1,1])
    return tf.nn.conv2d(inputTensor4D, kgx, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _sobley(inputTensor4D, padding='VALID'):
    """ Sobel grady filter 3x3 """
    sobely = [[ 1, 2, 1],
              [ 0, 0, 0],
              [-1,-2,-1]]
    kgy = tf.reshape(tf.constant(sobely, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgy, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _grady_3x3(inputTensor4D, padding='VALID'):
    # central difference 3x3, uniform grid
    gy = [[ 0, 1, 0],
          [ 0, 0, 0],
          [ 0,-1, 0]]
    gy = (1/2.0) * np.array(gy)
    kgy = tf.reshape(tf.constant(gy, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgy, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _grady_5x5(inputTensor4D, padding='VALID'):
    # central difference 5x5, uniform grid
    gy = [[0, 0, 1, 0, 0],
          [0, 0,-8, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 8, 0, 0],
          [0, 0, 1, 0, 0]]
    gy = (1/12.0) * np.array(gy)
    kgy = tf.reshape(tf.constant(gy, tf.float32), [5,5,1,1])
    return tf.nn.conv2d(inputTensor4D, kgy, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _sobelxx(inputTensor4D, padding='VALID'):
    """ Sobel gradxx (second-derivative) filter 3x3 """
    sobelxx = [[ 1,-2, 1],
               [ 2,-4, 2],
               [ 1,-2, 1]]
    kgxx = tf.reshape(tf.constant(sobelxx, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgxx, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _sobelyy(inputTensor4D, padding='VALID'):
    """ Sobel gradyy (second-derivative) filter 3x3 """
    sobelyy = [[ 1, 2, 1],
               [-2,-4,-2],
               [ 1, 2, 1]]
    kgyy = tf.reshape(tf.constant(sobelyy, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgyy, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _sobelxy(inputTensor4D, padding='VALID'):
    """ Sobel gradxy (second-derivative) filter 3x3 """
    sobelxy = [[ 1, 0,-1],
               [ 0, 0, 0],
               [-1, 0, 1]]
    kgxy = tf.reshape(tf.constant(sobelxy, tf.float32), [3,3,1,1])
    return tf.nn.conv2d(inputTensor4D, kgxy, strides=[1,1,1,1], padding=padding)

  @staticmethod
  def _laplacian(inputTensor4D, padding='VALID'):
    """ Sobel Laplacian filter 3x3 """
    lap_flt = [[ 0, 1, 0],
               [ 1,-4, 1],
               [ 0, 1, 0]]
    lap = tf.reshape(tf.constant(lap_flt, tf.float32), [3, 3, 1, 1])
    return tf.nn.conv2d(inputTensor4D, lap, strides=[1,1,1,1], padding=padding)


# New method to make symmetric kernel - generate smoother kernels.
def _makeSymConvKern(inputKernel4D, mode='XYTU'):
  """Make symmetric kernel over X-, Y-, Diagoanl-, Off-Diagonal axes, or any combination
  of these. The 4D kernel tensor shape must be [height, width, in_channel, out_channel]

  Parameters
  ----------
  inputKernel4D: 4D kernel tensor
  mode: string. 'X', 'Y', 'T', 'U', or any combination of these such as 'XY', 'TU', ...

  Returns
  -------
  ksym: symmetric 4D kernel tensor
  """

  # construct symmetric kernels
  def _make3DSymKern(inputKernel3D, axis):
    """Make a symmetric 3D kerenl by adding mirror symmetric kernel for a given axis.

    Parameters
    ----------
    inputKernel3D: 3D kernel tensor with shape = [height, width, channels]
    axis : string - a mode to represent symmetric operation axis
                    'X' = X-axis symmetric,  0.5*( k + k.x )
                    'Y' = Y-axis symmetric,  0.5*( k + k.y )
                    'T' = Diagonal-axis symmetric,  0.5*( k + k.T )
                    'U' = Off-diagonal-axis symmetric, 0.5*(rot90(k) + rot90(k).T)

    Returns
    -------
    symmetric 3D kernel tensor
    """
    k_in, k_mirror = None, None
    # X- or Y-axis mirror symmetric op
    if axis == 'X' or axis == 'Y' :
      flip_3D_image = tf.image.flip_left_right if axis=='X' else tf.image.flip_up_down
      k_in = inputKernel3D
      k_mirror = flip_3D_image(k_in)
    # Diagonal axis mirror symmetric op
    elif axis == 'T' :
      k_in = inputKernel3D
      k_mirror = tf.transpose(k_in, perm=[1,0,2])
    # Off-diagonal axis mirror symmetric op
    elif axis == 'U' :
      k_in = tf.image.rot90 (inputKernel3D)
      k_mirror = tf.transpose(k_in, perm=[1,0,2])
    else :
      raise ValueError ("Unknown option axis={0}".format(axis))

    k_sym = tf.stack ([k_in, k_mirror], axis=-1)
    return tf.multiply(0.5, tf.reduce_sum(k_sym, axis=-1))

  # tf kernel shape = [height, width, in_channel, out_channel]
  # input 3D image for tf.image.flip_* = [height, width, channels]  (tf.version 1.1)
  # unpack kernel over in-channel
  ki_unpack = tf.unstack (inputKernel4D, axis=2)
  ksym = []
  modes = list(mode)
  for ki in ki_unpack :
    for m in modes :
      ki = _make3DSymKern (ki, axis=m)
    ksym.append(ki)
  ksym = tf.stack(ksym, axis=2)
  #modelLogger.debug ("Constructed sym{0} kern, shape = {1}".format(mode, list(ksym.shape.as_list())))
  return ksym


# ---------------------------------------------------------
# Custom Layer for Conv2D using symmetric kernels in keras
# ---------------------------------------------------------
class SymConv2D(keras.layers.Conv2D):
  def __init__(self, **kwargs):
    super(SymConv2D, self).__init__(**kwargs)
    #self.mylayer = keras.layers.Lambda(function=_makeSymConvKern)

  def call(self, inp):
    # overwrite the call method
    self.sym_kern = _makeSymConvKern(self.kernel, mode='XYTU')
    #self.sym_kern = self.mylayer(self.kernel)
    actf_name = self.get_config()['activation']
    output = tf.nn.conv2d(inp, self.sym_kern, strides=[1,1,1,1], padding='VALID')
    if self.use_bias:
      output = tf.nn.bias_add(output, self.bias)
    if actf_name=='linear':
      return output
    elif actf_name=='sigmoid':
      return tf.nn.sigmoid(output)
    elif actf_name=='relu':
      return tf.nn.relu(output)
    else:
      return None


def plot_images(images, nrows=2, ncols=5):
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,4), sharex=True, sharey=True)
  axes = axes.flatten()
  for i, img in enumerate(images):
    axes[i].imshow(img)


def get_model_kernels(model):
  print(model.layers)
  kernels = []
  for ly in model.layers[1:-1]:
    #_kern = ly.kernel
    _kern = ly.sym_kern
    print(_kern)
    print(tf.shape(_kern))
    h, w, ic, oc = _kern.shape.as_list()
    _kern = tf.reshape(_kern, [h, w, ic*oc])
    _kern = tf.transpose(_kern, perm=[2, 0, 1])
    kernels.append(_kern)
  return kernels



def image_pool(batch=10, pixel=51, grain=4, show_images=False):
  # virtual data gen for unit_test
  # can check the symmetric kernel conv2d working correctly or not
  sigma = np.exp(np.random.rand(batch, 2)+3)
  tp = np.linspace(-grain*pixel//2, grain*pixel//2, pixel)
  X, Y = np.meshgrid(tp, tp)
  images = np.array([np.exp(-(X/sx)**2 -(Y/sy)**2 -(X*Y/(sx*sy))) for (sx, sy) in sigma])
  print(images.shape)
  if show_images:
    plot_images(images)
  return np.expand_dims(images, axis=-1)


def build_model(ksizes, channels):
  x = keras.Input(shape=[None, None, 1])
  outputs = [x]

  for k, c in zip(ksizes, channels):
    _y = SymConv2D(
      kernel_size=k, 
      filters=c, 
      activation='sigmoid', 
      kernel_regularizer=keras.regularizers.L2(1e-4))(outputs[-1])
    outputs.append(_y)

  y_final = keras.layers.Conv2D(kernel_size=1, filters=1, name='final')(outputs[-1])
  model = keras.Model(inputs=x, outputs=y_final)
  return model


def unit_test():
  input_images = image_pool(batch=100)
  _, img_size, _, _ = input_images.shape
  ksizes = [11,11,11]
  channels = [2, 2, 2]
  target_size = img_size - np.sum(ksizes) + len(ksizes)
  center = img_size//2
  s1 = center-target_size//2
  s2 = center+target_size//2 + 1
  target_images = input_images[:, s1:s2, s1:s2, :]
  model = build_model(ksizes, channels)
  model.summary()
  #model.compile(optimizer='adam', loss='mse')

  #model_kernels = get_model_kernels(model)
  #print(model_kernels[0])
  #return 0
  #for mk in model_kernels:
  #  fig, axes = plt.subplots(nrows=1, ncols=len(mk))
  #  for i in range(len(mk)):
  #    axes[i].imshow(mk[i])
  for ly in model.layers[1:-1]:
    print(ly)
    print(ly.sym_kern)
    skern = ly.sym_kern
    #skern = ly.kernel
    print(tf.shape(skern))

  return 0
  model.fit(
    x=input_images,
    y=target_images,
    batch_size=32,
    verbose=1,
    epochs=200,
    validation_split=0.2)

  #model_kernels = get_model_kernels(model)
  #for mk in model_kernels:
  #  fig, axes = plt.subplots(nrows=1, ncols=len(mk))
  #  for i in range(len(mk)):
  #    axes[i].imshow(mk[i])

  # prepare test images with different pixel size and check the model prediction (inference)
  test_images = image_pool(batch=5, pixel=73)
  results = model.predict(test_images)
  _, wi, _, _ = test_images.shape
  _, wf, _, _ = results.shape
  pad_width = (wi-wf)//2
  # plot the ground truth vs. prediction
  fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,5))
  for j in range(5):
    axes[0, j].imshow(test_images[j,:,:,0])
    infer_img = results[j,:,:,0]
    infer_img = np.pad(infer_img, pad_width=pad_width, mode='constant')
    axes[1, j].imshow(infer_img)


def main():
  #image_pool(show_images=True)
  unit_test()
  #plt.tight_layout()
  #plt.show()


if __name__=="__main__":
  main()

