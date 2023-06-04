import os, sys
import pandas as pd
import numpy as np
import argparse
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
import tensorflow as tf

# user-defined modules
import models, metrology, myutils


def get_net_code(k1, k2, c):
  return '-'+str(k1)+'.'+str(c)+'-'+str(k2)+'.'+str(c)+'-'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs', type=str, default='.npy')
  parser.add_argument('--targets', type=str, default='.npy')
  parser.add_argument('--asd_file', type=str, default='.asd')
  parser.add_argument('--learning_rate', type=float, default=0.0002)
  parser.add_argument('--l2rc', type=float, default=0.00001)
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--grain_size', type=float, default=8.0)
  parser.add_argument('--threshold', type=float, default=0.3988341)
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--kern1_size', type=int, default=21)
  parser.add_argument('--kern2_size', type=int, default=41)
  parser.add_argument('--show_step', type=int, default=10)
  parser.add_argument('--valid_step', type=int, default=50)
  parser.add_argument('--channels', type=int, default=4)
  parser.add_argument('--output_dir', type=str, default='train_outputs')

  FLAGS, _ = parser.parse_known_args()
  #net_code=get_net_code(FLAGS.ker1_size, FLAGS.kern2_size, FLAGS.channels)
  #workdir = 'cnn'+FLAGS.model_version+net_code+'L2REG'+str(FLAGS.l2rc)]
  if not os.path.isdir(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  
  myjob = Trainer(flags=FLAGS)
  myjob.data_preprocess()
  myjob.get_inputdata_info()
  #myjob.creating_batch()
  #myjob.train_loop()

  # He initializer 
  #zero_init = tf.zeros_initializer()
  #trainer(flags=FLAGS, initializer=he_init, workdir=workdir)



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
  print results
  return results['std'], results['mean']



def cnn2L(x, initializer, l2rc, flt1_shape, flt2_shape):
  h1, w1, ic1, oc1 = flt1_shape
  h2, w2, ic2, oc2 = flt2_shape
  myModel = models.CnnLayer(initializer, l2rc)
  Y1 = myModel.symCBA(x, flt1_shape, 'XYTU')
  Y2 = myModel.symCBA(Y1, flt2_shape, 'XYTU')
  modelY = myModel.CBA(Y2, [1,1,oc2,1], is_linear=True)
  return modelY


#TODO: cnn cross-terms, modify the network structure 
"""
def xnnV2(x, initializer, l2rc, flt1_shape, flt2_shape):
  h1, w1, ic1, oc1 = flt1_shape
  h2, w2, ic2, oc2 = flt2_shape
  myModel = models.CnnLayer(initializer, l2rc)
  Y1_basic = myModel.symCBA(x, flt1_shape, 'XYTU')
  conv_terms = tf.unstack(Y1_basic, axis=-1)
  #print conv_terms

  conv_Xterms=[]
  for i in range(1,oc1):
    temp = map(lambda x: tf.multiply(*x), zip(conv_terms, conv_terms[i:]))
    conv_Xterms.extend(temp)  
  #temp_lst = reduce(lambda x,y: x+y, [zip(conv_terms, conv_terms[i]) for i in range(1, oc1)])
  
  #print conv_Xterms
  all_terms = conv_terms + conv_Xterms
  Y1 = tf.stack(all_terms, axis=-1)
  flt2_update = [h2, w2, len(all_terms), 1]
  modelY = None
  if h2==1 and w2==1:
    modelY = myModel.CBA(Y1, flt2_update, is_linear=True)
  else:
    modelY = myModel.symCBA(Y1, flt2_update, 'XYTU', is_linear=True)
  return modelY
"""

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
    print "loading input data {}".format(flags.inputs)
    print "loading target data {}".format(flags.targets)
    self.dataX = np.load(flags.inputs)
    self.dataY = np.load(flags.targets)
    self.num_imgs, self.input_h, self.input_w = self.dataX.shape
    print "input raw data shape {}".format(self.dataX.shape)
    print "target raw data shape {}".format(self.dataY.shape)
    self.num_batches = self.num_imgs // self.batch_size
    self.gauge_index = np.arange(self.num_imgs)
    # calc output image size after 2 conv2d layers
    self.output_h = self.input_h - (self.kern1_size-1) - (self.kern2_size-1)
    print "output image size after 2 conv2d = {}".format(self.output_h)

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
      print "filter too big, no enough output image size"
      return

  def get_inputdata_info(self):
    # input and target information
    self.hgs, self.vgs, self.dfA = metrology._getGaugeInfo(self.asd)
    self.input_tslices = metrology._getTslices(np.squeeze(self.dataX_croped), self.hgs, self.vgs)
    self.target_tslices = metrology._getTslices(np.squeeze(self.dataY_croped), self.hgs, self.vgs)
    self.tarCD, unresolved_t = metrology._extractCDFromCenter(
      self.target_tslices, self.thresh, self.grain_size)
    self.dfA['target_CD']=self.tarCD   ; # NTD model CD (images as target)
    print self.dfA.head()

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
      print v.name, v.shape

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
          print "Epoch {}, loss_all={}, loss_MAE={}, loss_L2={}".format(
            i, tot_loss, vloss_mae, vloss_L2)
          loss_list.append((i, tot_loss, vloss_L2))

        if i>0 and i% self.valid_step==0:
          sess.run(self.valid_Iter, 
            feed_dict={self.inputPH:self.dataX, self.outputPH:self.dataY_croped})
          model_images, kern1_val, kern2_val = sess.run([self.modelY, kern1, kern2])
          #print model_images.shape
          output_tslices = metrology._getTslices(np.squeeze(model_images), self.hgs, self.vgs)
          modCD, unresolved_m = metrology._extractCDFromCenter(
            output_tslices, self.thresh, self.grain_size)
          print "found {} unresolved gauges".format(len(unresolved_m))
          vstd, vmean = calc_fiterr(modCD, self.tarCD)
          self.dfA['cnnsim_CD']=modCD
          RMS_list.append((i, vstd))  
      # end of training loop 
    # end of TF session
    dfRMS = pd.DataFrame(RMS_list, columns=["step", "std"])
    #dfRMS.to_csv(os.path.join(workdir, "std_results.csv"), sep=" ", index=False)
    #dfloss = pd.DataFrame(loss_list, columns=['step', 'total_loss', 'L2_loss'])
    #dfloss.to_csv(os.path.join(workdir, "loss_results.csv"), sep=" ", index=False)
    print dfRMS.head(10)
  
  """
  # print model QoR
  dfA['cnnsim_CD']=modCD
  dfA['fiterr_cnnsim2target']=dfA['cnnsim_CD']-dfA['target_CD']
  dfA['fiterr_cnnsim2wafer']=dfA['cnnsim_CD']-dfA['wafer_CD']
  dfA['fiterr_target2wafer']=dfA['target_CD']-dfA['wafer_CD']
  NTDmodel_QoR = dfA['fiterr_target2wafer'].agg(["std", "max", "min", "mean"])
  cnnsim2target_QoR = dfA['fiterr_cnnsim2target'].agg(["std", "max", "min", "mean"])
  cnnsim2wafer_QoR = dfA['fiterr_cnnsim2wafer'].agg(["std", "max", "min", "mean"])
  print "="*10+ " NTD model QoR (to wafer) " + "="*10
  print NTDmodel_QoR
  print "="*10+ " CNN simulator QoR (to wafer) " + "="*10
  print cnnsim2wafer_QoR
  print "="*10+ " CNN simulator QoR (to target) " + "="*10
  print cnnsim2target_QoR
  
  
  #plot_images_slices(dataX_croped, dataY_croped, model_images)
  #plot_conv_kernels(kern1val_ini, kern1val_final)
  #plt.show()
  """

  

if __name__=="__main__":
  main()
