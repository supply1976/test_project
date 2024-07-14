import os, sys, logging, argparse
import numpy as np
import pandas as pd
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
#tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')


def _makeSymConvKern(inputKernel4D, mode='XYTU'):
    """
    #New method to make symmetric kernel - generate smoother kernels.
    Make symmetric kernel over X-, Y-, Diagoanl-, Off-Diagonal axes, or any combination
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
    return ksym


# ---------------------------------------------------------
# Custom Layer for Conv2D using symmetric kernels in keras
# ---------------------------------------------------------
class SymConv2D(keras.layers.Conv2D):
    def __init__(self, **kwargs):
        super(SymConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        # overwrite the call method
        self.sym_kernel = _makeSymConvKern(self.kernel, mode='XY')
        outputs = tf.keras.backend.conv2d(
            inputs, 
            self.sym_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs - tf.keras.backend.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs


class DenseCropSkip(keras.layers.Layer):
    # using Dense() layer and Cropping2D() layer to do skip-connection
    def __init__(self, cropping, channel, regL2=1.0e-6, **kwargs):
        super().__init__(**kwargs)
        self.cropping = cropping
        self.channel = channel
        self.regL2 = regL2
        self.crop2D = keras.layers.Cropping2D(cropping=cropping)
        self.dense = keras.layers.Dense(
            units=channel, activation=None, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(regL2))

    def call(self, inputs):
        input_crop = self.crop2D(inputs)
        output = self.dense(input_crop)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
        "cropping": self.cropping, 
        "channel": self.channel,
        "regL2": self.regL2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TimeCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.time = []
        self.timetaken = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.time.append((epoch, np.around(time.time()-self.timetaken, 3)))

    def on_train_end(self, logs={}):
        for _ in self.time: print(_)


def calc_spline_values(x, grid, spline_order):
    """
    modified from https://github.com/ZPZhou-lab/tfkan/blob/main/tfkan/ops/spline.py
    also reference to Wiki about B-Splines definition

    Calculate B-spline values for the input tensor.

    Parameters
    ----------
    x : tf.Tensor
        The input nD tensor, with shape = [batch, h, w, ..., c]
    grid : tf.Tensor
        The grid 1D tensor with length (grid_intervals + 2 * spline_order + 1).
    spline_order : int
        The spline order.

    Returns: tf.Tensor
        B-spline bases tensor (n+1)D 
        shape = [batch, h, w, , ..., c, grid_intervals + spline_order].

    Example: 
    if x is a 4D input Tensor, 
    spline_order=3, grid_intervals=5, 
    grid =  [x0, x1, x2, x3, ..., x5]  (5+1 points)
    grid extension = [x_{-3}, x_{-2}, x_{-1}, x0, x1, ..., x5, x6, x7, x8] (6+3*2 points)
    
    calc a set of basis splines: {Bi(x)} ; i = -1, 0, 1, ..., 6  (8 spline functions)
    output is 5D Tensor with shape = [batch, h, w, c, 8]
    code can be implemented simply by broadcasting
    """
    assert len(tf.shape(x)) >= 2
    
    # grid reshape for broadcasting
    inp_dim = len(tf.shape(x))
    grid = tf.reshape(grid, [-1]+[1]*inp_dim)

    # init the order-0 B-spline bases
    bases = tf.logical_and(tf.greater_equal(x, grid[:-1]), tf.less(x, grid[1:]))
    bases = tf.cast(bases, x.dtype)
    
    # iter to calculate the B-spline values
    for k in range(1, spline_order + 1):
        bases_1 = (x-grid[: -(k+1)]) / (grid[k:-1] - grid[: -(k+1)]) * bases[:-1]
        bases_2 = (grid[(k+1):] - x) / (grid[(k+1):] - grid[1:(-k)]) * bases[1:]
        bases = bases_1 + bases_2
    bases = tf.stack(tf.unstack(bases), axis=-1)
    return bases


def bspline3_gaus_approx(x, grid):
    """
    Gaussian approximation of B-spline order 3 (under uniform grid)
    x:    tf.Tensor, input nD tensor, shape = [batch, h, w, ..., c] 
    grid: tf.Tensor, uniform grid 1D tensor
    h: grid resolution

    output: (n+1)D tensor, shape = [batch, h, w, ..., c, len(grid)-4] 
    """
    x = tf.expand_dims(x, axis=-1)
    x = x - grid[2:-2]
    arr = grid.numpy()
    h = arr[1]-arr[0]
    sigma = h / 1.7
    bases = 0.667 * tf.math.exp(-0.5 * ((x-2*h)**2/sigma**2))
    return bases


class BSplineACTF(keras.layers.Layer):
    """
    trainable activation functions using parameterized B-spline functions

    theory (inspired by KAN paper):
    phi(x) -- unknown nonlinear activation funciton to be trained

    orig: 
    phi(x) = c0*B3_0(x) + c1*B3_1(x) + c2*B3_2(x) + ...
    ci are trainable weights
    {Bp_i} for i = 0, 1, 2, ... is the set of B-Spline functions of order p 
    
    approx:
    B-Splinee order 3 function can be well-approximated by bspline3_gaus_approx(), 
    which is easy and fast

    input:  nD Tensor, shape = [batch, h, w, ..., c]
    output: nD Tensor, shape = [batch, h, w, ..., c]

    grid_range: default = (0, 1.0), 
        bounded range for trainable activation funciton, 
        outside this range, the output is decay to zero
    grid_intervals: int, how many intervals within the grid_range
    spline_order: int, default = 3
    use_gaus_approx: default is True, for order 3 B-splines, use Gaussian is much faster

    if depthwised is False: 
        output = phi(input), single fucntion phi() apply to all elements of input Tensor

    if depthwised is True: 
        different activation funcitons apply to differnt input channels.
    """
    def __init__(self, 
                 depthwised=False,
                 grid_range=(0.0,1.0),
                 grid_intervals=8, 
                 spline_order=3,
                 use_gaus_approx=True, 
                 **kwargs):
        super(BSplineACTF, self).__init__(**kwargs)
        self.depthwised = depthwised
        self.grid_range = grid_range
        self.grid_intervals = grid_intervals
        self.spline_order = spline_order
        self.use_gaus_approx = use_gaus_approx
        # do grid extentsion for accuracy (see KAN paper section 2.4)
        xL, xR = grid_range
        xL = xL - spline_order * (xR-xL)/self.grid_intervals
        xR = xR + spline_order * (xR-xL)/self.grid_intervals
        nums_grid = self.grid_intervals + 2*self.spline_order + 1
        self.grid = tf.constant(np.linspace(xL, xR, nums_grid), dtype=tf.float32)
        
    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.in_channel = in_channel
        # use a list of Dense layer (units=1, no bias) as the trainable coef of B-Splines
        self.denses = [
            keras.layers.Dense(units=1, use_bias=False) for _ in range(in_channel)]

    def call(self, inputs, *args, **kwargs):
        """
        num_basis = grid_intervals + spline_order
        inputs:    nD Tensor, shape = [batch, h, w, ..., c]
        Ax:    (n+1)D Tensor, shape = [batch, h, w, ..., c, num_basis]
        t:         nD Tensor, shape = [batch, h, w, ..., num_basis]
        t_out:     nD Tensor, shape = [batch, h, w, ..., 1]
        output:    nD Tensor, shape = [batch, h, w, ..., c]
        """
        if self.use_gaus_approx:
            Ax = bspline3_gaus_approx(inputs, self.grid)
        else:
            Ax = calc_spline_values(inputs, self.grid, self.spline_order)
        
        if self.depthwised:
            # unstack on channel axis
            tlist = tf.unstack(Ax, axis=-2)
            output =[]
            for i, t in enumerate(tlist):
                t_out = self.denses[i](t)
                output.append(t_out)
            output = tf.concat(output, axis=-1)
        else:
            output = self.denses[0](Ax)
            output = tf.reduce_mean(output, axis=-1)
        return output

    def get_config(self):
        config = super(BSplineACTF, self).get_config()
        config.update({
            "depthwised": self.depthwised,
            "grid_range": self.grid_range,
            "grid_intervals": self.grid_intervals,
            "spline_order": self.spline_order, 
            "use_gaus_approx": self.gaus_approx,
            })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(
    img_size, 
    kerns, 
    chans, 
    skips, 
    actfs, 
    dilas,
    enable_trainable_actfs,
    depthwised,
    grid_range,
    grid_intervals,
    spline_order=3,
    use_gaus_approx=True,
    ):
    assert len(kerns)==len(chans)
    assert len(kerns)==len(skips)
    assert len(kerns)==len(actfs)

    if dilas is None:
        dilas = [1]*len(kerns)
    effective_kerns = [k+(k-1)*(d-1) for (k, d) in zip(kerns, dilas)]
    layerID = range(len(kerns))

    # 1-channel input (for simple prototype test)
    x = keras.Input(shape=[img_size, img_size, 1])
    outputs=[x]
    for i, (k, c, d, s, a) in enumerate(zip(kerns, chans, dilas, skips, actfs)):
        conv_layer = SymConv2D(
            kernel_size=k, 
            filters=c, 
            dilation_rate=d, 
            activation=None, 
            use_bias=True,
            kernel_regularizer=keras.regularizers.l2(1.0e-6),
            name="symconv_"+str(i))
        
        if enable_trainable_actfs and i!=layerID[-1]:
            actf_layer = BSplineACTF(
                depthwised=depthwised,
                grid_range=grid_range, 
                grid_intervals=grid_intervals,
                spline_order=spline_order,
                use_gaus_approx=use_gaus_approx,
                name='bspline_actf_'+str(i),
                )
        else:
            actf_layer = keras.layers.Activation(a)
        
        y = conv_layer(outputs[-1])
        y = actf_layer(y)
        
        outputs.append(y)
        
        # handle the skips
        if s == -1:
            continue
        else:
            assert s >= 0
            inp_layerID = s
            out_layerID = i+1
            inp_tensor = outputs[inp_layerID]
            out_tensor = outputs[out_layerID]
            cropping = sum(effective_kerns[inp_layerID:out_layerID])-(out_layerID-inp_layerID)
            cropping = cropping // 2
            inp_tensor_crop = DenseCropSkip(cropping, chans[i])(inp_tensor)
            out_tensor_new = keras.layers.Add()([out_tensor, inp_tensor_crop])
            outputs[i+1] = out_tensor_new
    # 
    model = keras.Model(inputs=x, outputs=outputs[-1])
    return model


def resize_to_model_output(model, inputs, labels):
    _, h, w, c = inputs.shape
    assert h==w
    input_shape = (h, w, c)
    output_shape = model.compute_output_shape((1, *input_shape)).as_list()[1:]
    output_size = output_shape[0]
    _, label_size, _, _ = labels.shape
    if output_size < label_size:
        c = (label_size-output_size)//2
        labels = labels[:, c:-c, c:-c, :]
    return labels


def result_plots(imagesA, imagesB):
    _, hA, _, _ = imagesA.shape
    _, hB, _, _ = imagesB.shape
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15,5))
    for i in range(5):
        axes[0, i].imshow(imagesA[i])
        axes[1, i].imshow(imagesB[i])
        axes[2, i].plot(imagesA[i, hA//2, :, 0], '-o')
        axes[2, i].plot(imagesB[i, hB//2, :, 0], '-o')


def netconfig():
    # UNet66 (large kernel CNN BKM)
    """
    KERNELS =  [19,41,23,37,29,31, 1, 1, 1, 1, 1, 1]
    CHANNELS = [ 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8, 1]
    DILATIONS= [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    SKIPS    = [-1, 0,-1, 2,-1, 4, 5, 4, 3, 2, 1,-1]
    ACTIVATIONS = ['softplus']*6 + ['linear']*6
    """

    # resnet753-lite (small kernel CNN BKM)
    #"""
    KERNELS =  [ 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 1]
    #CHANNELS = [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1]
    CHANNELS = [16,16,16,16,16,16, 8, 8, 8, 8, 8, 8, 1]
    DILATIONS= [ 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 1]
    SKIPS    = [-1,-1, 0,-1,-1, 3,-1,-1, 6,-1,-1, 9, 0]
    ACTIVATIONS = ['relu']*12 + ['linear']
    #"""

    # resnetL21
    """
    KERNELS =  [19,21,21,19,21,21,21,21,21, 1]
    CHANNELS = [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    DILATIONS= None
    SKIPS    = [-1,-1, 0,-1,-1, 3,-1,-1, 6, 0]
    ACTIVATIONS = ['relu']*9 + ['linear']*1
    """

    return (KERNELS, CHANNELS, DILATIONS, SKIPS, ACTIVATIONS)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbfile', type=str, default=None)
    parser.add_argument('--testdbfile', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="my_trained_MLmodels")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--enable_trainable_actfs', action='store_true')
    parser.add_argument('--depthwise_actfs', action='store_true')

    FLAGS, _ = parser.parse_known_args()
    
    if not os.path.isdir(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu_id)

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)>0:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    
    # get network config
    kerns, chans, dilas, skips, actfs = netconfig()
    grid_range = (0.0, 1.0)
    grid_intervals = 8
    spline_order = 3

    cnnModel = build_model(
        img_size=None, 
        kerns=kerns, 
        chans=chans, 
        skips=skips, 
        actfs=actfs,
        dilas=dilas,
        enable_trainable_actfs=FLAGS.enable_trainable_actfs,
        depthwised=FLAGS.depthwise_actfs,
        grid_range=grid_range,
        grid_intervals=grid_intervals,
        )
    
    cnnModel.summary()

    # TODO
    # A bug when restoring checkpoint if optimizer is using Adam
    # see details and workaround in 
    # https://github.com/tensorflow/tensorflow/issues/33150
    # https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
    #opt = keras.optimizers.Adam(
    #    learning_rate=tf.Variable(FLAGS.lr),
    #    beta_1=tf.Variable(0.9),
    #    beta_2=tf.Variable(0.999),
    #    epsilon=tf.Variable(1e-7))
    #opt.iterations
    #opt.decay = tf.Variable(0.0)
    
    opt = keras.optimizers.Adam(learning_rate=FLAGS.lr)
    cnnModel.compile(loss='mse', optimizer=opt)

    model_save_path = os.path.join(FLAGS.save_dir, "best_ckpt")
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    try:
        plot_model(cnnModel, to_file="./model_graph.png", 
            show_shapes=True, show_layer_names=True)
    except:
        pass
   
    if FLAGS.restore:
        load_status = cnnModel.load_weights(model_save_path)
        load_status.assert_consumed()

    if FLAGS.dbfile is None:
        print("use fake data for training flow test")
        input_shape = (255, 255, 1)
        inputs = np.random.rand(100, *input_shape).astype(np.float32)
        labels = np.random.rand(100, *input_shape).astype(np.float32)

    elif FLAGS.dbfile.endswith('npz'):
        data = np.load(FLAGS.dbfile, allow_pickle=True)
        print(list(data.keys()))
        inputs = data['inputs']
        labels = data['labels']
        if len(inputs.shape)==3:
            inputs = np.expand_dims(inputs, axis=-1)
        if len(labels.shape)==3:
            labels = np.expand_dims(labels, axis=-1)
        
    else:
        print("{} not support yet".format(FLAGS.dbfile))
        return 0

    if FLAGS.testdbfile is not None:
        testdata = np.load(FLAGS.testdbfile)
        images = testdata['images']
        Presist, vt0, _ = list(np.transpose(images, [3, 0, 1, 2]))
        test_inputs = np.expand_dims(vt0, axis=-1)
        test_labels = np.expand_dims(Presist, axis=-1)
    else:
        cid = np.random.choice(len(inputs), 5, replace=False)
        test_inputs = inputs[cid]
        test_labels = labels[cid]
   
    labels = resize_to_model_output(cnnModel, inputs, labels)
    test_labels = resize_to_model_output(cnnModel, test_inputs, test_labels)
    print(inputs.shape, labels.shape)
    print(test_inputs.shape, test_labels.shape)
   
    if FLAGS.training:
        t0 = time.time()
        cnnModel.fit(
            x=inputs,
            y=labels,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            verbose=1,
            validation_split=0.2,
            callbacks=[ckpt_callback],
            )
        deltaT = time.time() - t0

        print("training completed")
        print("input={}, output={}, time elapsed={} seconds".format(
            inputs.shape, labels.shape, np.round(deltaT, 2)))
        print("model weights are saved")

    else:
        print("No Training")
    
    # trained actf visualize
    if FLAGS.enable_trainable_actfs:
        names = [lyr.name for lyr in cnnModel.layers if "bspline_actf" in lyr.name]
        tractf_wts = [lyr.weights for lyr in cnnModel.layers if "bspline_actf" in lyr.name]
        tractf_wts = [np.concatenate([x.numpy() for x in a], axis=-1) for a in tractf_wts]
        xsamples = np.linspace(-1.0, 2.0, 100)
        x = tf.expand_dims(tf.constant(xsamples, tf.float32), axis=0)
        a, b = grid_range
        a = a - spline_order * (b-a)/grid_intervals
        b = b + spline_order * (b-a)/grid_intervals
        grid = tf.constant(np.linspace(a, b, grid_intervals+7), tf.float32)
        #Ax = calc_spline_values(x, grid, spline_order=3)
        Ax = bspline3_gaus_approx(x, grid)
        Ax = Ax.numpy()[0]
        _, c = tractf_wts[0].shape

        fig, axes = plt.subplots(nrows=c, ncols=len(tractf_wts), sharey=True, figsize=(18,3))
        if c==1:
            axes = axes.reshape([1, -1])
        for i, wt in enumerate(tractf_wts):
            _, col = wt.shape
            val = np.matmul(Ax, wt)
            for j in range(col):
                axes[j, i].plot(xsamples, val[:, j], '-')
                axes[j, i].grid()
            axes[0, i].set_title(names[i])

    y_pred = cnnModel.predict(test_inputs)
    y_true = test_labels
    result_plots(y_true, y_pred)



main()
plt.show()
    
