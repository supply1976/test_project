from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


# load multiple image(jpeg) files with tensorflow
def load_image(file_list, channels=1):
    with tf.name_scope('image_loader'):
        # make a queue from a list of filenames
        filename_quene = tf.train.string_input_producer(file_list, shuffle=True)
        # a reader instance to read entire image files
        reader = tf.WholeFileReader()
        # the first return of reader.read() is filename which we don't need
        _, image_content = reader.read(filename_quene)
        # decode the JPEG file, 
        # return to a 3D Tensor of type uint8 with shape [height, width, channels]
        image = tf.image.decode_jpeg(image_content, channels=channels) 
    return image


def mini_batch(image_orig):
    batch_size = 32
    num_preprocess_threads = 1
    min_queue_examples = 256
    
    image=tf.image.resize_images(image_orig,[500,500])
    
    images = tf.train.shuffle_batch([image], 
        batch_size=batch_size, 
        num_threads=num_preprocess_threads, 
        capacity=min_queue_examples + 3 * batch_size, 
        min_after_dequeue=min_queue_examples)
    return images


# preprocess for su, make low resolution image and ground truth
def lr_gen(image, input_size, output_size, channels=3, scale=2, batch_size=1):
    lr_images=[]
    ground_truths=[]
    for i in range(batch_size):
        cropped_image = tf.random_crop(image, [input_size, input_size, channels])
        ground_truth = tf.image.resize_images(cropped_image, (output_size, output_size))
        resized_image = tf.image.resize_images(cropped_image, (input_size//scale, input_size//scale), 
                                 method=tf.image.ResizeMethod.BICUBIC)
        lr_image = tf.image.resize_images(resized_image, (input_size, input_size),
                            method=tf.image.ResizeMethod.BICUBIC)
        ground_truths.append(ground_truth)
        lr_images.append(lr_image)
        # stack a list of 3D Tensors to a 4D Tensor    
    lr_images = tf.stack(lr_images, axis=0)
    ground_truths = tf.stack(ground_truths, axis=0)
    # 4D Tensor of type float with shape [batch_size, height, width, channels]
    return (lr_images/255, ground_truths/255)


def main():
    # only looking for .jpg files in the image_dir, not scan files in sub-dir
    image_dir = '/Users/tacoWu/tfApp/srcnn_proj1/images/'
    trees = os.walk(image_dir)
    _, _, files = trees.next()
    jpgfiles = [os.path.join(image_dir, fn) for fn in files if fn.endswith('.jpg')]
    
    image = load_image(jpgfiles, channels=3)
    #lr_images, ground_truths = lr_gen(image, 100, 100, batch_size=1)
    images = mini_batch(image)
    
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    
    print images
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(coord=coord)
        print threads
        img = sess.run(images)
        
        print img.shape
        coord.request_stop()
        coord.join(threads)

    #plt.imshow(img)
    #plt.show()
    
if __name__ == '__main__':
    main()

