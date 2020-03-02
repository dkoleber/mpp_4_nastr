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

        # train_labels = tf.one_hot(train_labels, 10)
        # test_labels = tf.one_hot(test_labels, 10)

        # train_labels_1h = np.zeros((train_labels.size, 10))
        # test_labels_1h = np.zeros((test_labels.size, 10))
        #
        # train_labels_1h[np.arange(train_labels.size), train_labels] = 1
        # test_labels_1h[np.arange(test_labels.size), test_labels] = 1
        #
        # train_labels = train_labels_1h
        # test_labels = test_labels_1h

        train_images = np.true_divide(train_images, 127.5)
        train_images = train_images - 1.
        test_images = np.true_divide(test_images, 127.5)
        test_images = test_images - 1.

        # train_images = np.true_divide(train_images, 255.)
        # test_images = np.true_divide(test_images, 255.)

        return Dataset(train_images, train_labels, test_images, test_labels)

    @staticmethod
    def get_build_set() -> Dataset:
        train_images = np.zeros([4, 16, 16, 3])
        train_labels = np.zeros([4, 1])

        test_images = np.zeros([4, 16, 16, 3])
        test_labels = np.zeros([4, 1])

        return Dataset(train_images, train_labels, test_images, test_labels)