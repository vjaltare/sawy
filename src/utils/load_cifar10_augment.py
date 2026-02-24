"""
Load and augment CIFAR-10 dataset. 
Augmentation includes:
- random cropping
- random horizontal flipping
- normalization
- random brightness adjustment
- random contrast adjustment
"""

import jax
import math
import jax.numpy as jnp
from typing import Callable
from functools import partial

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
from tensorflow.keras import layers

DATA_PATH = "/local_disk/vikrant/datasets"

def load_cifar10_augment(
        batch_size: int,
        train_steps: int,
        seed: int,
        shuffle_buffer: int,
        augmentation: bool = True,
        training: bool = False,
        data_dir: str = DATA_PATH
):
    # set the seed
    tf.random.set_seed(seed)

    # define the augmentation layers
    data_augmentation = tf.keras.Sequential(
        [
        #     layers.ZeroPadding2D(padding=4),
        #     layers.RandomCrop(32, 32),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=[-0.05, 0.05]),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # ±10% zoom
            layers.RandomBrightness(factor=[-0.2, 0.2]),
            layers.RandomContrast(factor=0.2),
        ], name="data_augmentation",
    )

    # normalization layer
    normalization = layers.Rescaling(1./255)

    # load the dataset
    train_ds = tfds.load('cifar10', split='train', data_dir=data_dir, shuffle_files=True)
    valid_ds = tfds.load('cifar10', split='test[:50%]', data_dir=data_dir)
    test_ds = tfds.load('cifar10', split='test[50%:]', data_dir=data_dir)

    # Extract images and labels
    def extract_image_label(example):
        return example['image'], example['label']
    
    train_ds = train_ds.map(extract_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(extract_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(extract_image_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation to training set
    if augmentation:
        def augment_train(image, label):
            # Cast to float32
            image = tf.cast(image, tf.float32)
            # Apply augmentation pipeline
            image = data_augmentation(image, training=True)
            # Normalize to [0, 1]
            image = normalization(image)
            return image, label
        
        train_ds = train_ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Just normalize without augmentation
        train_ds = train_ds.map(
            lambda img, lbl: (normalization(tf.cast(img, tf.float32)), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Normalize validation and test sets (no augmentation)
    def normalize_only(image, label):
        image = tf.cast(image, tf.float32)
        image = normalization(image)
        return image, label
    
    valid_ds = valid_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(shuffle_buffer, seed=seed)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.repeat()
    train_ds = train_ds.take(train_steps)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Convert to dict format for compatibility
    def to_dict_format(image, label):
        return {'image': image, 'label': label}
    
    train_ds = train_ds.map(to_dict_format, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(to_dict_format, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(to_dict_format, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_ds, valid_ds, test_ds










