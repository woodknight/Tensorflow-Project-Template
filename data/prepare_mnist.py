"""
This file is for preparing mnist and extracting it from the binary files

# Please first download cifar100 dataset and extract it in data folder here!!
# Then run this script to prepare the data of cifar100

- Generates numpys
- Generates images
- Generates tfrecords
"""
import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.datasets import mnist

def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def save_imgs_to_disk(path, arr, file_names):
    for i, img in tqdm(enumerate(arr)):
        imageio.imwrite(path + file_names[i], img, 'PNG-PIL')

def save_numpy_to_disk(path, arr):
    np.save(path, arr)


def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and transposing the numpy array of the images
    x_train = x_train.reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28, 28))

    x_train_len = x_train.shape[0]
    x_test_len = x_test.shape[0]

    print(x_train.shape)
    print(x_train.dtype)
    print(y_train.shape)
    print(y_train.dtype)
    print(x_test.shape)
    print(x_test.dtype)
    print(y_test.shape)
    print(y_test.dtype)

    if not os.path.exists('./data/mnist/imgs/train/'):
        os.makedirs('./data/mnist/imgs/train/')
    if not os.path.exists('./data/mnist/imgs/test/'):
        os.makedirs('./data/mnist/imgs/test/')

    x_train_filenames = []
    x_test_filenames = []
    for i in range(x_train_len):
        x_train_filenames.append(str(i) + '.png')
    for i in range(x_test_len):
        x_test_filenames.append(str(i) + '.png')

    # Save the filename of x_train and y_train
    with open('data/mnist/x_train_filenames.pkl', 'wb') as f:
        pickle.dump(x_train_filenames, f)
    with open('data/mnist/x_test_filenames.pkl', 'wb') as f:
        pickle.dump(x_test_filenames, f)

    print("FILENAMES OF IMGS saved successfully")

    print("Saving the imgs to the disk..")

    save_imgs_to_disk('./data/mnist/imgs/train/', x_train, x_train_filenames)
    save_imgs_to_disk('./data/mnist/imgs/test/', x_test, x_test_filenames)

    print("IMGS saved successfully")

    print("Saving the numpys to the disk..")

    save_numpy_to_disk('./data/mnist/x_train.npy', x_train)
    save_numpy_to_disk('./data/mnist/y_train.npy', y_train)
    save_numpy_to_disk('./data/mnist/x_test.npy', x_test)
    save_numpy_to_disk('./data/mnist/y_test.npy', y_test)

    print("Numpys saved successfully")

    print("Saving the data numpy pickle to the disk..")

    # SAVE ALL the data with one pickle
    with open('data/mnist/data_numpy.pkl', 'wb')as f:
        pickle.dump({'x_train': x_train,
                     'y_train': y_train,
                     'x_test': x_test,
                     'y_test': y_test,
                     }, f)

    print("DATA NUMPY PICKLE saved successfully..")

    print('saving tfrecord..')

    save_tfrecord_to_disk('data/mnist/train.tfrecord', x_train, y_train)
    save_tfrecord_to_disk('data/mnist/test.tfrecord', x_test, y_test)

    print('tfrecord saved successfully..')


if __name__ == '__main__':
    main()
