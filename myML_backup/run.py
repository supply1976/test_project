import os, sys
import argparse
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# user-defined modules
import metrology


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
    def __init__(self, mode, **kwargs):
        super(SymConv2D, self).__init__(**kwargs)
        self.mode = mode

    def call(self, inp):
        # overwrite the call method
        self.sym_kern = _makeSymConvKern(self.kernel, mode=self.mode)
        actf_name = self.get_config()['activation']
        padding = self.padding.upper()
        output = tf.nn.conv2d(inp, self.sym_kern, strides=[1,1,1,1], padding=padding)
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


def get_net_code(k, c):
    k = map(str, k)
    c = map(str, c)
    net_code = [".".join(i) for i in zip(k, c)]
    net_code = "-".join(net_code)
    return net_code


def resNetBlock(
    kernels:list, 
    channels:list, 
    ksym_mode:str,
    regL2:float, 
    actf):

    assert len(kernels)==len(channels)
    
    def apply(inputs):
        """
        inputs: 4D image tensor: [batch, h, w, c]
        """
        inp_ch = inputs.shape[3]
        out_ch = channels[-1]
        _pad = (sum(kernels)-len(kernels)+1)//2
        # create delta-function 4D kernel
        kdelta = np.ones([1, 1, inp_ch, out_ch])
        kdelta = np.pad(kdelta, pad_width=[(_pad,_pad), (_pad,_pad), (0,0), (0,0)])
        kdetla = tf.constant(kdelta, tf.float32)
        # conv with delta kernel to resize inputs with correct (height, width) 
        x = tf.nn.conv2d(inputs, kdelta, strides=[1,1,1,1], padding='VALID', name="x")
        
        residual = inputs 
        for (k, c) in zip(kernels, channels):
            residual = SymConv2D(
                mode=ksym_mode, 
                kernel_size=k, 
                filters=c, 
                kernel_regularizer=keras.regularizers.L2(regL2))(residual)
            residual = actf(residual)
        res_out = keras.layers.Add()([x, residual])
        return res_out
    return apply


def build_model(img_size, regL2, actf):
    image_input = keras.Input(shape=(img_size, img_size, 1), name="image_input")
    x = resNetBlock([21, 41], [4, 2], 'XY', regL2, actf)(image_input)
    x = resNetBlock([31, 51], [4, 2], 'XY', regL2, actf)(x)
    x = resNetBlock([31, 31, 1], [2, 4, 1], 'XY', regL2, actf)(x)
    model = keras.Model(inputs=image_input, outputs=x, name="mynet")
    return model



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--database', type=str, default=None)
  parser.add_argument('--asd', type=str, default=None)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--l2', type=float, default=0.0001)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--grain_size', type=float, default=8.0)
  parser.add_argument('--threshold', type=float, default=0.3988341)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--kernels', nargs='+', type=int, default=[21, 31, 41])
  parser.add_argument('--channels', nargs='+', type=int, default=[2, 2, 2])
  parser.add_argument('--show_step', type=int, default=10)
  parser.add_argument('--valid_step', type=int, default=50)
  parser.add_argument('--output_dir', type=str, default='train_outputs')

  FLAGS, _ = parser.parse_known_args()
  net_code = get_net_code(FLAGS.kernels, FLAGS.channels)
  workdir = 'cnn-'+net_code

  if not os.path.isdir(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  
  model = build_model(
          img_size=205,
          regL2=0.0001,
          actf=keras.activations.sigmoid)

  model.summary()
  


def save_infer_result_to_npz(workdir, x, yc, ym, t0, t1, t2, k1i, k2i, k1f, k2f, dfA, epoch):
  # save results
  results={}
  results['valid_step']=np.array(epoch)
  results['input_images']=x
  results['target_images']=yc
  results['cnnsim_images']=ym
  results['input_tslices']=t0
  results['target_tslices']=t1
  results['cnnsim_tslices']=t2
  results['kern1_ini']=k1i
  results['kern2_ini']=k2i
  results['kern1_final']=k1f
  results['kern2_final']=k2f
  results['asd'] = np.array(dfA.to_dict(orient='list'))
  output_fn = os.path.join(workdir, 'epoch_'+str(epoch).zfill(5)+'.npz')
  np.savez(output_fn, **results)


def calc_fiterr(modelCD, targetCD):
  d={}
  d['modelCD']=modelCD
  d['targetCD']=targetCD
  d['fiterr']=modelCD - targetCD
  df = pd.DataFrame(d)
  results = df['fiterr'].agg(["std", "mean", "max", "min"])
  return results['std'], results['mean']



def cnn2L(x, initializer, l2rc, flt1_shape, flt2_shape):
  h1, w1, ic1, oc1 = flt1_shape
  h2, w2, ic2, oc2 = flt2_shape
  myModel = models.CnnLayer(initializer, l2rc)
  Y1 = myModel.symCBA(x, flt1_shape, 'XYTU')
  Y2 = myModel.symCBA(Y1, flt2_shape, 'XYTU')
  modelY = myModel.CBA(Y2, [1,1,oc2,1], is_linear=True)
  return modelY



class Trainer(object):
  def __init__(self, flags=None):
    self.show_step = flags.show_step
    self.valid_step = flags.valid_step
    # hyper-parameters
    self.lr = flags.learning_rate
    self.l2rc = flags.l2rc
    self.epochs = flags.epochs
    self.batch_size = flags.batch_size
    # metrology 
    self.thresh = flags.threshold
    self.asd = flags.asd_file
    self.grain_size = flags.grain_size
    # network parameters
    self.he_init= tf.contrib.layers.variance_scaling_initializer()
    self.kern1_size = flags.kern1_size
    self.kern2_size = flags.kern2_size
    self.channels = flags.channels
    self.flt1 = [self.kern1_size, self.kern1_size, 1, self.channels]
    self.flt2 = [self.kern2_size, self.kern2_size, self.channels, self.channels]
    # load raw data (.npy)
    #print "loading input data {}".format(flags.inputs)
    #print "loading target data {}".format(flags.targets)
    self.dataX = np.load(flags.inputs)
    self.dataY = np.load(flags.targets)
    self.num_imgs, self.input_h, self.input_w = self.dataX.shape
    #print "input raw data shape {}".format(self.dataX.shape)
    #print "target raw data shape {}".format(self.dataY.shape)
    self.num_batches = self.num_imgs // self.batch_size
    self.gauge_index = np.arange(self.num_imgs)
    # calc output image size after 2 conv2d layers
    self.output_h = self.input_h - (self.kern1_size-1) - (self.kern2_size-1)
    #print "output image size after 2 conv2d = {}".format(self.output_h)

  def data_preprocess(self):
    # expand to 4D np array for TF conv2d
    self.dataX = np.expand_dims(self.dataX, axis=-1)
    self.dataY = np.expand_dims(self.dataY, axis=-1)
    self.dataX_croped = None
    self.dataY_croped = None
    if self.output_h >10:
      sd = (self.input_h - self.output_h)//2 
      self.dataX_croped = self.dataX[:, sd:-sd, sd:-sd, :]
      self.dataY_croped = self.dataY[:, sd:-sd, sd:-sd, :]
    else:
      #print "filter too big, no enough output image size"
      return

  def get_inputdata_info(self):
    # input and target information
    self.hgs, self.vgs, self.dfA = metrology._getGaugeInfo(self.asd)
    self.input_tslices = metrology._getTslices(np.squeeze(self.dataX_croped), self.hgs, self.vgs)
    self.target_tslices = metrology._getTslices(np.squeeze(self.dataY_croped), self.hgs, self.vgs)
    self.tarCD, unresolved_t = metrology._extractCDFromCenter(
      self.target_tslices, self.thresh, self.grain_size)
    self.dfA['target_CD']=self.tarCD   ; # NTD model CD (images as target)
    #print self.dfA.head()

  def creating_batch(self):
    # tf.dataset batcher
    self.inputPH = tf.placeholder(tf.float32, [None, self.input_h, self.input_h, 1])
    self.outputPH = tf.placeholder(tf.float32, [None, self.output_h, self.output_h, 1])
    self.dataset = tf.data.Dataset.from_tensor_slices((self.inputPH, self.outputPH))
    self.dataset = self.dataset.shuffle(buffer_size=1000)
    self.dataset = self.dataset.repeat().batch(self.batch_size)
    self.dataset_val = tf.data.Dataset.from_tensor_slices((self.inputPH, self.outputPH))
    self.dataset_val = self.dataset_val.batch(self.num_imgs)
    self.iterator = tf.data.Iterator.from_structure(
      self.dataset.output_types, self.dataset.output_shapes)
    #iterator = dataset.make_initializable_iterator()
    #iterator = dataset.make_one_shot_iterator()
    self.batch_x, self.batch_y = self.iterator.get_next()

  def train_loop(self): 
    self.modelY = cnn2L(self.batch_x, self.he_init, self.l2rc, self.flt1, self.flt2)
    loss_L2 = tf.add_n(tf.get_collection('costs_L2'))
    loss_mae = tf.reduce_mean(abs(self.modelY-self.batch_y))
    # Binary Cross Entropy (BCE)  
    #labels = tf.nn.sigmoid(batch_y-thresh)
    #logits = modelY-thresh
    #sig_bce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    #loss_bce = tf.reduce_mean(sig_bce)
    #loss = loss_L2 + loss_bce
    loss = loss_L2 + loss_mae

    opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
    init = tf.global_variables_initializer() 
    # print all variables
    _vars = tf.global_variables()
    for v in _vars:
      print(v.name, v.shape)

    self.train_Iter = self.iterator.make_initializer(self.dataset)
    self.valid_Iter = self.iterator.make_initializer(self.dataset_val)

    with tf.Session() as sess:
      sess.run(init)
      kern1 = [v for v in sess.graph.get_collection('variables') if v.name=='kern1:0'][0]
      kern2 = [v for v in sess.graph.get_collection('variables') if v.name=='kern2:0'][0]
      # before training
      kern1_inival, kern2_inival = sess.run([kern1, kern2])

      RMS_list=[]
      loss_list=[]
      # start the training loop
      for i in range(self.epochs+1):
        sess.run(self.train_Iter, 
          feed_dict={self.inputPH:self.dataX, self.outputPH:self.dataY_croped}) 
        tot_loss = 0
        for _ in range(self.num_batches):
          # batch loss
          _, vloss, vloss_mae, vloss_L2 = sess.run([opt, loss, loss_mae, loss_L2])
          tot_loss += vloss
        tot_loss = tot_loss/(self.num_imgs//self.batch_size)

        if i % self.show_step ==0:
          #print "Epoch {}, loss_all={}, loss_MAE={}, loss_L2={}".format(
          #i, tot_loss, vloss_mae, vloss_L2)
          loss_list.append((i, tot_loss, vloss_L2))

        if i>0 and i% self.valid_step==0:
          sess.run(self.valid_Iter, 
            feed_dict={self.inputPH:self.dataX, self.outputPH:self.dataY_croped})
          model_images, kern1_val, kern2_val = sess.run([self.modelY, kern1, kern2])
          #print model_images.shape
          output_tslices = metrology._getTslices(np.squeeze(model_images), self.hgs, self.vgs)
          modCD, unresolved_m = metrology._extractCDFromCenter(
            output_tslices, self.thresh, self.grain_size)
          #print "found {} unresolved gauges".format(len(unresolved_m))
          vstd, vmean = calc_fiterr(modCD, self.tarCD)
          self.dfA['cnnsim_CD']=modCD
          RMS_list.append((i, vstd))  
      # end of training loop 
    # end of TF session
    dfRMS = pd.DataFrame(RMS_list, columns=["step", "std"])
    #dfRMS.to_csv(os.path.join(workdir, "std_results.csv"), sep=" ", index=False)
    #dfloss = pd.DataFrame(loss_list, columns=['step', 'total_loss', 'L2_loss'])
    #dfloss.to_csv(os.path.join(workdir, "loss_results.csv"), sep=" ", index=False)
    #print dfRMS.head(10)
  
  

if __name__=="__main__":
  main()
