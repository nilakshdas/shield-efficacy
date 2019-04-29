import sys

import keras
import tensorflow as tf

sys.path.append('differentiable-jpeg')
from utils import differentiable_jpeg

# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L157
resnet50_keras_preprocessing_fn = keras.applications.resnet50.preprocess_input


def differentiable_slq(x, qualities=(20, 40, 60, 80), patch_size=8):
    num_qualities = len(qualities)

    with tf.name_scope('DifferentiableSLQ'):
        one = tf.constant(1, name='one')
        zero = tf.constant(0, name='zero')

        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]

        patch_n = tf.cast(n / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)
        patch_m = tf.cast(m / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.image.resize_nearest_neighbor(
            [tf.random_uniform(
                (patch_n, patch_m, 3),
                0, num_qualities, dtype=tf.int32)],
            (patch_n * patch_size, patch_m * patch_size),
            name='random_layer_indices')[0, :, :, 0][:n, :m]
        indices = tf.transpose(
            tf.stack([Z, R, C]),
            perm=[1, 2, 0],
            name='random_layer_indices')

        x_expanded = tf.expand_dims(x, 0, name='expanded_image')
        x_compressed_stack = tf.stack(list(map(
            lambda q: tf.squeeze(differentiable_jpeg(x_expanded, quality=q), [0]),
            qualities)), name='compressed_images')

        x_slq = tf.gather_nd(x_compressed_stack, indices, name='final_image')

    return x_slq


def slq(x, qualities=(20, 40, 60, 80), patch_size=8):
    num_qualities = len(qualities)

    with tf.name_scope('SLQ'):
        x = tf.cast(tf.cast(x, tf.int32), tf.uint8)

        one = tf.constant(1, name='one')
        zero = tf.constant(0, name='zero')

        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]

        patch_n = tf.cast(n / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)
        patch_m = tf.cast(m / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.image.resize_nearest_neighbor(
            [tf.random_uniform(
                (patch_n, patch_m, 3),
                0, num_qualities, dtype=tf.int32)],
            (patch_n * patch_size, patch_m * patch_size),
            name='random_layer_indices')[0, :, :, 0][:n, :m]
        indices = tf.transpose(
            tf.stack([Z, R, C]),
            perm=[1, 2, 0],
            name='random_layer_indices')

        x_compressed_stack = tf.stack(
            list(map(
                lambda q: tf.image.decode_jpeg(tf.image.encode_jpeg(
                    x, format='rgb', quality=q), channels=3),
                qualities)),
            name='compressed_images')

        x_slq = tf.gather_nd(x_compressed_stack, indices, name='final_image')
        x_slq = tf.cast(x_slq, tf.float32)

    return x_slq
