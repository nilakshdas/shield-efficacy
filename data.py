from glob import glob
import os

import tensorflow as tf


def _decode_tf_example(serialized_example, image_size=224):
    """Parses and decodes a serialized tensorflow example.
    Args:
        serialized_example (tf.train.Example):
            A single serialized tensorflow example.
    Returns:
        dict: A dictionary mapping feature keys to tensors.
    """

    tf_resize = lambda x, s=image_size: tf.image.resize_bilinear(
        x, [s, s], align_corners=False)

    feature_set = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    }

    features = tf.parse_single_example(serialized_example, features=feature_set)

    image_id = features['image/filename']
    label = features['image/class/label']
    height, width = tf.cast(features['image/height'], tf.int32), \
                    tf.cast(features['image/width'], tf.int32)
    
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.reshape(image, [height, width, 3])
    image = tf.expand_dims(image, 0)
    image = tf_resize(image)
    image = tf.squeeze(image, [0])

    label = label - 1  # make labels 0-999 instead of 1-1000

    return image, label


def load_tfrecords_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_decode_tf_example)
    return dataset
