#!/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os, sys, re
import numpy as np
import tensorflow as tf
#from resistml_gconsts import *
#from profiler import CustomSummary
#import utils
#modelLogger = utils.getLogger('model')


#------------------------------------------------------------------------------------#
# Get variables from CPU shared among multiple GPUs.
# old implementation for TF <=1.7 only
#------------------------------------------------------------------------------------#
def _variableOnCPU(name, shape, initializer, constraint=None):
  with tf.device('/cpu:0') :
    try :
      # tf.VERSION >= 1.4
      var = tf.get_variable(
        name=name, 
        shape=shape, 
        initializer=initializer, 
        constraint=constraint)
    except TypeError :
      # tf.VERSION < 1.4
      var = tf.get_variable(name=name, shape=shape, initializer=initializer)
  return var


def _variableWithWeightDecay(name, shape, initializer, l2rc=None, constraint=None):
  var = _variableOnCPU(name, shape, initializer, constraint)
  if l2rc is not None :
    weight_decay = tf.multiply(tf.nn.l2_loss(var), l2rc, name='weight_loss')
    tf.add_to_collection('costs_L2', weight_decay)
  return var


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
def _makeSymConvKern(inputKernel4D, mode):
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


def _makeObliqSymConvKern(inputKernel4D):
  """
    Parameters
    ----------
    inputKernel4D : 4D tf tensor
                    kernal shape is [height, width, in_channels, out_channels]

    Returns
    -------
    4D tf tensor which is symmetric over 45 degrees
  """

  # make hamming window
  win_size = inputKernel4D.shape[0]
  hwin_1d = tf.contrib.signal.hamming_window (window_length=win_size)
  hwin_2d = tf.tensordot(hwin_1d, hwin_1d, axes=0)
  CustomSummary.Add (hwin_2d, 'hwin_2d', mode='save')
  hwin_4d = tf.expand_dims(tf.expand_dims(tf.sqrt(hwin_2d), axis=-1), axis=-1)

  # apply hamming window to inputKernel
  input_kern = hwin_4d * inputKernel4D

  # Rotate Kernel 45 degree
  angle = 0.25 * np.pi
  # change shape HWNC => NHWC
  obliq_kern = tf.transpose (input_kern, perm=[2,0,1,3])
  obliq_kern = tf.contrib.image.rotate(obliq_kern, angles=angle, interpolation='BILINEAR')
  # change shape NHWC => HWNC
  obliq_kern = tf.transpose (obliq_kern, perm=[1,2,0,3])

  symkern = 0.5 * (input_kern+ obliq_kern)
  return symkern


#------------------------------------------------------------------------------------#
# Layer Definitions
#------------------------------------------------------------------------------------#

class CnnLayer(object):
  """ Base class of the CNN Layer class. Create initial bias and kernel weights from
      _variableOnCPU () and _variableWithWeightDecay(). Basic layer operations are
      defined as symCBA (Symmetric Conv Bias Activation) and CBA (Conv Bias Activation).
  """

  def __init__(self, initializer, l2rc=None):
    self.initializer = initializer
    self.l2rc = l2rc
    self.layer_i = 0
    return

  def __getBiasKernNames(self):
    self.layer_i += 1
    layer_idx = str(self.layer_i)
    bias_name = 'bias{}'.format(layer_idx)
    kern_name = 'kern{}'.format(layer_idx)
    return bias_name, kern_name


  def symCBA(self, input_tensor, kern_shape, sym_dir, is_linear=False):
    """ Conv2d-Bias-Activation, as a combined single operation
        force the conv2d kernel to be mirror symmetric along X-, Y-, Diag-, Off-Diag axes.
        Args :
          input_tensor (4D tf tensor):
            input 4D tensor with shape [batch, height, width, channel]
          kern_shape (4D int array):
            convolution kernel with shape = [height, width, in_channel, out_channel]
          sym_dir (str): 
            one of 'X', 'Y', 'T', 'U', or any combinations of these
        Return :
          scba (4D tf tensor) : conv2d-bias-activated ouput with shape = [batch, height, width, channel]
    """
    is_45deg_sym = sym_dir[-1] == 'S'
    if is_45deg_sym :
      sym_dir = sym_dir[:-1]

    bias_shape = kern_shape[-1]
    bias_name, kern_name = self.__getBiasKernNames()

    bias =           _variableOnCPU(name=bias_name, shape=bias_shape, initializer=self.initializer)
    kern = _variableWithWeightDecay(name=kern_name, shape=kern_shape, initializer=self.initializer,
                                    l2rc=self.l2rc)
    kern_sym = _makeSymConvKern(kern, mode=sym_dir)

    if is_45deg_sym :
      modelLogger.debug("Perform 45 degree kernel symmetric operation")
      kern_sym = _makeObliqSymConvKern (kern_sym)

    # convolution + bias => activation
    conv = tf.nn.conv2d (input_tensor, kern_sym, strides=[1,1,1,1], padding='VALID')
    if is_linear:
      scba = tf.nn.bias_add(conv, bias)
    else:
      scba = tf.sigmoid (tf.nn.bias_add(conv, bias), name=None)

    # Save the symmetric kernel and bias
    #if self.save_kern_sym :
    #kern_name = 'SYM_' + kern_name
    #CustomSummary.Add(bias,     "INFERDATA/bias:{0}".format(bias_name), mode='save')
    #CustomSummary.Add(kern_sym, "INFERDATA/kernel:{0}".format(kern_name), mode='save')

    #modelLogger.debug ("symCBA-{0} (dir={1}) : output shape = {2}".format(self.layer_i, sym_dir, scba.shape))
    return scba


  def CBA(self, input_tensor, kern_shape, sym_dir='N', is_linear=False):  # sym_dir should be always 'N'
    """ Conv2d-Bias-Activation, as a combined single operation
        Same as symCBA (), but do not enforce any symmetries.
    """
    bias_shape = kern_shape[-1]
    bias_name, kern_name = self.__getBiasKernNames()

    bias =           _variableOnCPU (name=bias_name, shape=bias_shape, initializer=self.initializer)
    kern = _variableWithWeightDecay (name=kern_name, shape=kern_shape, initializer=self.initializer, l2rc=self.l2rc)

    # convolution + bias => activation
    conv = tf.nn.conv2d (input_tensor, kern, strides=[1,1,1,1], padding='VALID')
    if is_linear:
      cba = tf.nn.bias_add(conv, bias)
    else:
      cba = tf.sigmoid (tf.nn.bias_add(conv, bias), name=None)

    # Save the kernel and bias
    #CustomSummary.Add(bias, "INFERDATA/bias:{0}".format(bias_name), mode='save')
    #CustomSummary.Add(kern, "INFERDATA/kernel:{0}".format(kern_name), mode='save')

    #modelLogger.debug ("CBA-{0} : output shape = {1}".format(self.layer_i, cba.shape))
    return cba



class XCNN_NLS1(CnnLayer):
  """ Total N layers - N-1 hidden (symCBA), 1 scaling (CBA) layer """

  def __init__(self, layer_cfg, initializer, l2rc=None):
    CnnLayer.__init__(self,
                      initializer=initializer,
                      l2rc=l2rc)
    self.layer_cfg = layer_cfg


  def __call__(self, input_tensor):
    modelLogger.debug("XCNN_NLS1 : Construct CNN layer from input tensor shape = {0}".format(input_tensor.shape))
    xin, xout = input_tensor, None

    for sym_mode, kern_shape in self.layer_cfg.genKernShape() :
      layerfunc = self.symCBA if sym_mode != 'N' else self.CBA
      xout = layerfunc (xin, kern_shape, sym_dir=sym_mode)
      xin = xout

    modelLogger.debug ("XCNN_NLS1 : output shape = {0}".format(xout.shape))
    return xout

#------------------------------------------------------------------------------------#
# Model definitions
#------------------------------------------------------------------------------------#

class Deform(object):
  """ Deformation model. Assume that the amount of deformation is small in the final image (Sm)
      compared to the initial (S). That is, the final image is given by adding small displacement
      vector (u) to the initial image,
          Sm = S + grad(S) * u
      And, this displacement vector, u can be found by taking gradient of unknown scalar function
      psi. Therefore,
           Sm = S + grad(S)*grad(psi)
      where psi = CNN(S).
  """

  def __init__(self,
               input_tensor,
               layer_cfg,
               l2rc = None,
               initializer=tf.random_normal_initializer()):
    try :
      CNN = getattr(sys.modules[__name__], 'XCNN_NLS1')
    except AttributeError :
      raise AttributeError ("Failed to get CNN layer type")

    cnnLayer = CNN (layer_cfg, initializer, l2rc)
    with tf.variable_scope("unknown_field1") :
      self.displacementU = cnnLayer (input_tensor)

    total_filter_size = sum([ks for ks in layer_cfg.kern_sizes if ks != 1]) - layer_cfg.n_sym_layer
    s1 = total_filter_size // 2
    s2 = s1 + 1
    self.resized_input_toU = input_tensor[:, s1: -s1, s1:-s1, :]
    self.resized_input_fin = input_tensor[:, s2: -s2, s2:-s2, :]

    # grad_i (AI)
    self.Pgx = _gradx(self.resized_input_toU)
    self.Pgy = _grady(self.resized_input_toU)

    # grad_i (U) : displacement vector
    self.Pux = _gradx(self.displacementU)
    self.Puy = _grady(self.displacementU)
    modelLogger.debug ("Deform: displacementU graph has been constructed.")
    return


  def FirstOrder(self):
    """ First order of deformation : grad(S)*grad(psi) """
    # the amount of the deformed value in the first order
    modelLogger.debug ("Deform: First order deformation graph is constructed.")
    return tf.add (self.Pux*self.Pgx, self.Puy*self.Pgy, name='delta_signal_FO')


  def ILaplU(self):
    """ Laplacian term derived from continuity eq. :  S*laplacian(psi) """
    lapl_U = _laplacian(self.displacementU)
    modelLogger.debug ("Deform: Laplacian graph is constructed")
    return tf.multiply (lapl_U, self.resized_input_fin, name='delta_signal_ILU')


  def SecondOrder(self):
    """ Second order of deformation """

    # gradxx(I), gradyy(I), gradxy(I)
    Pgxx = _gradxx(self.resized_input_toU)
    Pgyy = _gradyy(self.resized_input_toU)
    Pgxy = _gradxy(self.resized_input_toU)

    # ux^2, uy^2, uxy
    Pux2 = tf.square(self.Pux)
    Puy2 = tf.square(self.Puy)
    Puxy = tf.multiply(self.Pux, self.Puy)

    Pgxx_ux2 = tf.multiply(Pgxx, Pux2)   # grad_xx(I) * ux^2
    Pgyy_uy2 = tf.multiply(Pgyy, Puy2)   # grad_yy(I) * uy^2
    Pgxy_uxy = tf.multiply(2.0, tf.multiply (Pgxy, Puxy))   # 2 * grad_xy(I) * uxy

    Pdelta_secondorder = tf.add_n([Pgxx_ux2, Pgyy_uy2, Pgxy_uxy])
    modelLogger.debug ("Deform: Second order deformation graph is constructed.")
    return tf.multiply(0.5, Pdelta_secondorder, name='delta_signal_SO')

  #---- Add your model, and define it in ModelGraph
  #def Unknown (self) :
  #  pass


  # the last tf variable name must be 'final_output'.
  def FinalOutput(self, delta_signal, delta_thresh=None):
    """ Return the final model image. Add all perturbation term and threshold """
    if delta_thresh is None :
      modelLogger.debug ("Deform: final_output graph is constructed without adjusting threshold.")
      return tf.add (self.resized_input_fin, delta_signal, name='final_output')
    else :
      output = tf.add (self.resized_input_fin, delta_signal)
      modelLogger.debug ("Deform: final_output graph is constructed with adjusting threshold.")
      return tf.add (output, delta_thresh, name='final_output')


def adjustThreshold(l2rc):
  thresh_constraint = lambda x : tf.clip_by_value(x, -0.1, 0.1)
  thresh = _variableWithWeightDecay ('thresh',
                                     None,
                                     initializer=tf.constant(0.0),
                                     l2rc=l2rc,
                                     constraint=thresh_constraint)
  CustomSummary.Add(thresh, 'thresh', mode='log')
  return thresh


#------------------------------------------------------------------------------------#
# Define the models, and set the model version
#------------------------------------------------------------------------------------#
def _getModelName(model_ver):
  """ Choose model name along the version """

  # Currently we support only one model form - deform
  if model_ver < "0.4" :
    modelname = 'Deform'
  else :
    raise RuntimeError ("Invalid model version {0}.".format(model_ver))
  return modelname


def ModelGraph(model_ver,
               thresh_adjust,
               input_tensor,
               layer_cfg,
               l2rc=None,
               initializer=tf.random_normal_initializer()
              ):
  """ According to the model version, choose model and get the final image.
      Args
        model_ver (str)      : model version
        thresh_adjust (bool) : If True, adjust const threshold, else, not.
        channels (int)       : number of channels.
        filter_size (int)    : Kernel filter size.
        l2rc (float)         : L2 Regularization weight.
        initializer          : Initializer for CNN kernels and biases.
      Return
        output (tf 4D tensor) : output model tensorflow graph
  """
  # Initialize model instance
  modelname = _getModelName (model_ver)
  modelAttr = getattr(sys.modules[__name__], modelname)
  modelInstance = modelAttr (input_tensor,
                             layer_cfg,
                             l2rc=l2rc,
                             initializer=initializer
                            )
  # delta signal and threshold
  delta_thresh = adjustThreshold (l2rc=l2rc) if thresh_adjust else None
  c1, c2 = None, None
  delta_signal = None

  # first order deformation
  if model_ver <= '0.1.9' :
    delta_signal = modelInstance.FirstOrder()
  # first order + I*LaplU
  elif model_ver <= '0.2.9' :
    if model_ver == '0.2.0' :
      delta_signal = tf.add(modelInstance.FirstOrder(), modelInstance.ILaplU())
    # first order + c1 * I*LaplU where c1 is a regression parameter
    elif model_ver == '0.2.1':
      c1_constraint = lambda x : tf.clip_by_value(x, -10.0, 10.0)
      c1 = _variableWithWeightDecay ('c1', None, initializer=tf.constant(-1.0),
                                     l2rc=l2rc, constraint=c1_constraint)
      if utils._IS_DEBUG_CNN_RESISTML_CAP :
        CustomSummary.Add (c1, 'c1', mode='log')
      delta_signal = tf.add(modelInstance.FirstOrder(), tf.multiply(c1, modelInstance.ILaplU()))
    elif model_ver == '0.2.2' :
      raise ValueError ("Not yet")
    else :
      raise ValueError ("Do not support model version {0}".format(model_ver))

  # second order deformation + I*LaplU
  elif model_ver >= '0.3.0' :
    if model_ver == '0.3.0' :
      delta_signal = tf.add(modelInstance.FirstOrder(), modelInstance.SecondOrder())
    # first order + c1*I*LaplU + c2*second_order, where c1 and c2 are regression parameters
    elif model_ver == '0.3.1' :
      c1_constraint = lambda x : tf.clip_by_value(x, -10.0, 10.0)
      c1 = _variableWithWeightDecay ('c1', None, initializer=tf.constant(-1.0),
                                     l2rc=l2rc, constraint=c1_constraint)
      if utils._IS_DEBUG_CNN_RESISTML_CAP :
        CustomSummary.Add (c1, 'c1', mode='log')
      delta_signal = tf.add(modelInstance.FirstOrder(), tf.multiply(c1, modelInstance.SecondOrder()))
    else :
      c1_constraint = lambda x : tf.clip_by_value(x, -10.0, 10.0)
      c2_constraint = lambda x : tf.clip_by_value(x, -10.0, 10.0)
      c1 = _variableWithWeightDecay ('c1', None, initializer=tf.constant(-1.0),
                                     l2rc=l2rc, constraint=c1_constraint)
      c2 = _variableWithWeightDecay ('c2', None, initializer=tf.constant(1.0),
                                     l2rc=l2rc, constraint=c2_constraint)

      if utils._IS_DEBUG_CNN_RESISTML_CAP :
        CustomSummary.Add (c1, 'c1', mode='log')
        CustomSummary.Add (c2, 'c2', mode='log')

      delta_signal = tf.add(modelInstance.FirstOrder(), tf.multiply(c1, modelInstance.ILaplU()))
      delta_signal = tf.add(delta_signal, tf.multiply(c2, modelInstance.SecondOrder()))

  output = modelInstance.FinalOutput (delta_signal, delta_thresh)

  # Save infer scalar variables
  scalar_variables = {'delta_thresh':delta_thresh,
                      'c1': c1,
                      'c2': c2
  }
  for key, val in scalar_variables.iteritems():
    if val is not None:
      CustomSummary.Add (val, 'INFERDATA/scalar:{0}'.format(key), mode='save')

  return output
