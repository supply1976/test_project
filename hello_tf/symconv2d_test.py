import os, sys, argparse
import math, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# use TF >= 2.4
import tensorflow as tf
from tensorflow import keras
#import tensorflow_addons as tfa
import tensorflow_datasets as tfds


def augment(img):
    """Flips an image left/right randomly."""
    return tf.image.random_flip_left_right(img)


def resize_and_rescale(img, size, clip_min=0.0, clip_max=1.0):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [clip_min, clip_max].

    Args:
        img: Image Tensor
        size: Desired image size for resizing
    Returns:
        Resized and rescaled image tensor
    """
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)

    img = tf.image.crop_to_bounding_box(img, 
        (height - crop_size)// 2,
        (width - crop_size) // 2,
        crop_size, 
        crop_size)

    # Resize
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)

    # Rescale the pixel values
    img = img / (255.0/(clip_max-clip_min) - clip_max)
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img


def _makeSymConvKern(inputKernel4D, mode='XYTU'):
    """
    inputKernel4D: [height, width, in_channels, out_channels]
    mode: string. 'X', 'Y', 'T', 'U', or combination of these such as 'XTYU'

    Return
    ------
    ksym: symmetric 4D kernel tensor
    """
    def _make3DsymKern(inputKernel3D, axis):
        k_in, k_mirror = None, None
        if axis=='X' or axis=='Y':
            flip_3D_image = tf.image.flip_left_right if axis=='X' else tf.image.flip_up_down
            k_in = inputKernel3D
            k_mirror = flip_3D_image(k_in)
        elif axis=='T':
            k_in = inputKernel3D
            k_mirror = tf.transpose(k_in, perm=[1,0,2])
        elif axis=='U':
            k_in = tf.image.rot90(inputKernel3D)
            k_mirror = tf.transpose(k_in, perm=[1,0,2])
        else:
            raise ValueError("Unkonwn option axis={0}".format(axis))

        k_sym = tf.stack([k_in, k_mirror], axis=-1)
        return tf.multiply(0.5, tf.reduce_sum(k_sym, axis=-1))

    # unpack kernel over in_channels
    ki_unpack = tf.unstack(inputKernel4D, axis=2)
    ksym = []
    modes = list(mode)
    for ki in ki_unpack:
        for m in modes:
            ki = _make3DsymKern(ki, axis=m)
        ksym.append(ki)
    ksym = tf.stack(ksym, axis=2)
    return ksym


class MySymConv2D(keras.layers.Conv2D):
    def __init__(self, **kwargs):
        super(MySymConv2D, self).__init__(**kwargs)

    def call(self, inp):
        self.sym_kern = _makeSymConvKern(self.kernel)
        padding = self.padding.upper()
        output = tf.nn.conv2d(inp, self.sym_kern, strides=[1,1,1,1], padding=padding)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output



def build_model():
    image_input = keras.layers.Input(shape=(None, None, 1), name="image_input")
    y = MySymConv2D(filters=2, kernel_size=11, use_bias=True)(image_input)
    y = keras.activations.sigmoid(y)
    y = MySymConv2D(filters=1, kernel_size=11, use_bias=True)(y)
    y = keras.activations.sigmoid(y)
    model = keras.Model(inputs=image_input, outputs=y)
    return model


def main():
    #ds = tfds.load("mnist", split="train")
    #ds = ds.map(lambda x: tf.cast(x['image'], dtype=tf.float32))
    (train_images, train_labels), _ = keras.datasets.mnist.load_data()
    train_images = np.expand_dims(train_images.astype(np.float32), axis=-1)
    train_images = train_images / 255.0

    targets = tf.convert_to_tensor(train_images, dtype=tf.float32)
    inputs = tf.image.resize(train_images, size=(48, 48), antialias=True)

    img_in = tf.clip_by_value(inputs[0:8]*255.0, 0.0, 255.0).numpy().astype(np.uint8)
    img_out = tf.clip_by_value(targets[0:8]*255.0, 0.0, 255.0).numpy().astype(np.uint8)
    
    model = build_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(x=inputs, y=targets, epochs=10)
    
    pred_images = model.predict(inputs[0:8])

    #test_images = np.stack([x.numpy() for x in train_ds[0].take(8)], axis=0)
    
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(10,5))
    for i in range(8):
        axes[0, i].imshow(img_in[i])
        axes[1, i].imshow(img_out[i])
        axes[2, i].imshow(pred_images[i])
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    pass

main()
plt.tight_layout()
plt.show()
