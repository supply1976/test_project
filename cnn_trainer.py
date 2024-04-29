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


class SkipConv(keras.layers.Layer):
    def __init__(self, cropping, channel):
        super().__init__()
        self.cropping = cropping
        self.channel = channel
        self.myCrop2D = keras.layers.Cropping2D(cropping=cropping)
        self.conv2D1x1 = keras.layers.Conv2D(filters=channel, kernel_size=1, 
            activation=None)

    def call(self, inputs):
        input_crop = self.myCrop2D(inputs)
        output = self.conv2D1x1(input_crop)
        return output


class TimeCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.time = []
        self.timetaken = time.process_time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.time.append((epoch, np.around(time.process_time()-self.timetaken, 3)))

    def on_train_end(self, logs={}):
        for _ in self.time: print(_)


def build_model(img_size, kerns, chans, skips, actfs, dilas=None):
    assert len(kerns)==len(chans)
    assert len(kerns)==len(skips)
    assert len(kerns)==len(actfs)
    if dilas is None:
        dilas = [1]*len(kerns)
    effective_kerns = [k+(k-1)*(d-1) for (k, d) in zip(kerns, dilas)]
    layerID = range(len(kerns))

    # 2-channel input
    x = keras.Input(shape=[img_size, img_size, 2])
    outputs=[x]
    for i, (k, c, d, s, a) in enumerate(zip(kerns, chans, dilas, skips, actfs)):
        _CBA = keras.layers.Conv2D(
            kernel_size=k, 
            filters=c, 
            dilation_rate=d, 
            activation=a, 
            name="CBA_"+str(i))
        y = _CBA(outputs[-1])
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
    num_images = 2174
    timetaken = TimeCallBack()
    """
    # UNet_new
    KERNELS =  [51,31,11,51,31,11, 1, 1, 1, 1, 1, 1, 1]
    CHANNELS = [ 2, 4, 2, 4, 8, 4, 4, 8, 4, 2, 4, 2, 1]
    DILATIONS= [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    SKIPS    = [-1,-1, 0,-1,-1, 3, 5, 4, 3, 2, 1, 0,-1]
    ACTIVATIONS = ['softplus']*12 + ['linear']
    """

    # UNet66
    #KERNELS =  [41,37,31,29,23,19, 1, 1, 1, 1, 1, 1]
    #KERNELS =  [19,23,29,31,37,41, 1, 1, 1, 1, 1, 1]
    KERNELS =  [19,41,23,37,29,31, 1, 1, 1, 1, 1, 1]
    CHANNELS = [ 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8, 1]
    DILATIONS= [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    SKIPS    = [-1, 0,-1, 2,-1, 4, 5, 4, 3, 2, 1,-1]
    ACTIVATIONS = ['softplus']*11 + ['linear']

    # resnet753 (small kernel CNN BKM)
    #KERNELS =  [ 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 1]
    #CHANNELS = [16,16,16,16,16,16, 8, 8, 8, 8, 8, 8, 1]
    #DILATIONS= [ 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 1]
    #SKIPS    = [-1,-1, 0,-1,-1, 3,-1,-1, 6,-1,-1, 9, 0]
    #ACTIVATIONS = ['softplus']*12 + ['linear']
    
    cnnModel = build_model(img_size, KERNELS, CHANNELS, SKIPS, ACTIVATIONS, DILATIONS)
    cnnModel.summary()
    cnnModel.compile(loss='mse', optimizer='adam')
    
    try:
        plot_model(cnnModel, to_file="./model_graph.png", show_shapes=True, show_layer_names=True)
    except:
        pass

    output_shape = cnnModel.compute_output_shape((1, img_size, img_size,2)).as_list()[1:]
    input_images = np.random.rand(num_images, img_size, img_size, 2)
    target_images = np.random.rand(num_images, *output_shape)
    
    t0 = time.time()
    cnnModel.fit(
        x=input_images, 
        y=target_images, 
        epochs=10, 
        batch_size=64, 
        verbose=1,
        callbacks=[timetaken])
    
    deltaT = time.time() - t0

    print("training time: ")
    print("  input shape = {}, output shape ={}, time elapsed={} seconds".format(
        input_images.shape, target_images.shape, np.round(deltaT, 2)))

main()

    
