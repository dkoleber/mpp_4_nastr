from __future__ import annotations
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.images_shape = train_images[0].shape
        self.labels_shape = test_labels[0].shape

    @staticmethod
    def get_cifar10() -> Dataset:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_labels = tf.one_hot(train_labels, 10)
        test_labels = tf.one_hot(test_labels, 10)
        return Dataset(train_images, train_labels, test_images, test_labels)

    @staticmethod
    def get_build_set() -> Dataset:
        train_images = np.zeros([4, 16, 16, 3])
        train_labels = np.zeros([4, 10])

        test_images = np.zeros([4, 16, 16, 3])
        test_labels = np.zeros([4, 10])

        return Dataset(train_images, train_labels, test_images, test_labels)