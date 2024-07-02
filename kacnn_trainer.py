import os, sys, logging, argparse
import numpy as np
import pandas as pd
import time
os.environ['CUDA_VISIBLE_DEVICE']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt


gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel('ERROR')

#tf.compat.v1.disable_eager_execution()



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
        self.sym_kernel = _makeSymConvKern(self.kernel, mode='XYTU')
        #actf_name = self.get_config()['activation']
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


class SkipConv(keras.layers.Layer):
    def __init__(self, cropping, channel, **kwargs):
        super().__init__(**kwargs)
        self.cropping = cropping
        self.channel = channel
        self.myCrop2D = keras.layers.Cropping2D(cropping=cropping)
        self.conv2D1x1 = keras.layers.Conv2D(
            filters=channel, kernel_size=1, activation=None,
            kernel_regularizer=keras.regularizers.l2(0.001))

    def call(self, inputs):
        input_crop = self.myCrop2D(inputs)
        output = self.conv2D1x1(input_crop)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"cropping": self.cropping, "channel": self.channel})
        return config


class TimeCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.time = []
        self.timetaken = time.process_time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.time.append((epoch, np.around(time.time()-self.timetaken, 3)))

    def on_train_end(self, logs={}):
        for _ in self.time: print(_)


def calc_spline_values(x, grid, spline_order):
    """
    modified from https://github.com/ZPZhou-lab/tfkan/blob/main/tfkan/ops/spline.py
    
    Calculate B-spline values for the input tensor.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor (nD), with shape = [batch, h, w, c, ...]
    grid : tf.Tensor
        The grid tensor (1D) with length (grid_size + 2 * spline_order + 1).
    spline_order : int
        The spline order.

    Returns: tf.Tensor
        B-spline bases tensor (n+1)D 
        shape = [batch, h, w, c, ..., grid_size + spline_order].
        B_ik === B_k(x_i)  
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


class BSplineACTF(keras.layers.Layer):
    def __init__(self, 
                 out_channel=None, 
                 grid_size=5, 
                 spline_order=3,
                 grid_range=(-1.0,1.0), **kwargs):
        super(BSplineACTF, self).__init__(**kwargs)
        self.out_channel = out_channel
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        #self.kernel_shape = (self.spline_basis_size, self.input_channel)
        
    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.in_channel = in_channel
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] - self.grid_range[0]

        # build grid (extention)
        self.grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size,
            self.grid_range[1] + self.spline_order * bound / self.grid_size,
            self.grid_size + 2*self.spline_order +1)

        self.grid = tf.Variable(
            initial_value=tf.cast(self.grid, dtype=tf.float32),
            trainable=False,
            dtype=tf.float32,
            name="spline_grd")

        if self.out_channel is None:
            kernel_shape = (self.spline_basis_size, self.in_channel)
        else:
            assert isinstance(self.out_channel, int)
            kernel_shape = (self.spline_basis_size, self.in_channel, self.out_channel)

        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=kernel_shape,
            initializer=tf.keras.initializers.RandomNormal(0.1),
            trainable=True,
            dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        Bik = calc_spline_values(inputs, self.grid, self.spline_order)
        if self.out_channel is None:
            spline_out = tf.einsum("...ik,ki->...i", Bik, self.spline_kernel)
        else:
            spline_out = tf.einsum("...ik,kij->...j", Bik, self.spline_kernel)
        return spline_out

    def get_config(self):
        config = super(DenseKAN, self).get_config()
        config.update({
            "out_channel": self.out_channel,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order, 
            "grid_range": self.grid_range})


def build_model(
    img_size, 
    kerns, 
    chans, 
    skips, 
    actfs, 
    dilas=None, 
    trainable_actfs=False,
    grid_size=5,
    spline_order=3,
    grid_range=(-1.0,1.0),
    out_channel=None
    ):
    assert len(kerns)==len(chans)
    assert len(kerns)==len(skips)
    assert len(kerns)==len(actfs)

    if dilas is None:
        dilas = [1]*len(kerns)
    effective_kerns = [k+(k-1)*(d-1) for (k, d) in zip(kerns, dilas)]
    layerID = range(len(kerns))

    # 1-channel input
    x = keras.Input(shape=[img_size, img_size, 1])
    outputs=[x]
    for i, (k, c, d, s, a) in enumerate(zip(kerns, chans, dilas, skips, actfs)):
        conv_layer = SymConv2D(
            kernel_size=k, 
            filters=c, 
            dilation_rate=d, 
            activation=None, 
            #kernel_regularizer=keras.regularizers.l2(0.001),
            name="symconv_"+str(i))
        if trainable_actfs and i != layerID[-1]:
            actf_layer = BSplineACTF(
                out_channel=out_channel,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range)
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
            inp_tensor_crop = SkipConv(cropping, chans[i])(inp_tensor)
            out_tensor_new = keras.layers.Add()([out_tensor, inp_tensor_crop])
            outputs[i+1] = out_tensor_new
    # 
    model = keras.Model(inputs=x, outputs=outputs[-1])
    return model


def main():
    img_size = 205
    num_images = 1000  ; #98689
    timetaken = TimeCallBack()

    """
    # UNet66
    KERNELS =  [19,41,23,37,29,31, 1, 1, 1, 1, 1, 1]
    CHANNELS = [ 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8, 1]
    DILATIONS= [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    SKIPS    = [-1, 0,-1, 2,-1, 4, 5, 4, 3, 2, 1,-1]
    ACTIVATIONS = ['softplus']*6 + ['linear']*6
    
    # resnet753 (small kernel CNN BKM)
    KERNELS =  [ 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 1]
    CHANNELS = [16,16,16,16,16,16, 8, 8, 8, 8, 8, 8, 1]
    DILATIONS= [ 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 1]
    SKIPS    = [-1,-1, 0,-1,-1, 3,-1,-1, 6,-1,-1, 9, 0]
    ACTIVATIONS = ['relu']*12 + ['linear']
    """
    # resnetL21
    KERNELS =  [21,21, 1,21,21, 1,21,21, 1, 1]
    CHANNELS = [ 4, 4, 4, 4, 4, 4, 4, 4,16, 1]
    DILATIONS= [ 2, 1, 1, 2, 1, 1, 2, 1, 1, 1]
    SKIPS    = [-1,-1, 0,-1,-1, 3,-1,-1, 6, 0]
    ACTIVATIONS = ['relu']*9 + ['linear']*1

    #cnnModel = build_model(img_size, KERNELS, CHANNELS, SKIPS, ACTIVATIONS, DILATIONS)
    cnnModel = build_model(
        img_size, KERNELS, CHANNELS, SKIPS, ACTIVATIONS, DILATIONS, 
        trainable_actfs=True)
    cnnModel.summary()
    cnnModel.compile(loss='mse', optimizer='adam')
    
    #configs = cnnModel.get_config()
    #for layer_config in configs['layers']:
    #    for k in layer_config.keys():
    #        print(k, layer_config[k])
    
    try:
        plot_model(cnnModel, to_file="./model_graph.png", show_shapes=True, show_layer_names=True)
    except:
        pass
    
    #out_size = img_size - sum(KERNELS) + len(KERNELS)
    #output_shape = [out_size, out_size, 1]
    output_shape = cnnModel.compute_output_shape((1, img_size, img_size, 1)).as_list()[1:]
    input_images = np.random.rand(num_images, img_size, img_size, 1).astype(np.float32)
    target_images = np.random.rand(num_images, *output_shape).astype(np.float32)
    print(input_images.dtype, target_images.dtype)
    
    t0 = time.time()
    cnnModel.fit(
        x=input_images, 
        y=target_images, 
        epochs=10, 
        batch_size=64, 
        verbose=1,
        validation_split=0.2,
        callbacks=[timetaken])
    
    deltaT = time.time() - t0

    print("training time: ")
    print("  input shape = {}, output shape ={}, time elapsed={} seconds".format(
        input_images.shape, target_images.shape, np.round(deltaT, 2)))


main()

    
