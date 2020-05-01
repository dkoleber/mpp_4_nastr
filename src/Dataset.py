from __future__ import annotations
import numpy as np
import tensorflow as tf
import os
from FileManagement import *

class ImageDataset:
    def __init__(self, images, labels, train_percentage, test_percentage, validation_percentage):
        num_samples = labels.shape[0]

        self.images_shape = images[0].shape
        self.labels_shape = labels[0].shape
        self.num_train_samples = int(num_samples * train_percentage)
        self.num_test_samples = int(num_samples * test_percentage)
        self.num_validation_samples = num_samples - (self.num_test_samples + self.num_train_samples)

        self.images = images
        self.labels = labels

        self.train_images = self.images[0:self.num_train_samples, :, :, :]
        self.test_images = self.images[self.num_train_samples: self.num_train_samples + self.num_test_samples, :, :, :]
        self.validation_images = self.images[-self.num_validation_samples:, :, :, :]

        self.train_labels = self.labels[0:self.num_train_samples, :]
        self.test_labels = self.labels[self.num_train_samples: self.num_train_samples + self.num_test_samples, :]
        self.validation_labels = self.labels[-self.num_validation_samples:, :]

        self.shuffle()

    def shuffle(self):
        pass  # TODO

    @staticmethod
    def get_cifar10() -> ImageDataset:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

        images = np.concatenate((train_images, test_images))
        labels = np.concatenate((train_labels, test_labels))

        images = np.true_divide(images, 127.5)
        images = images - 1.

        return ImageDataset(images, labels, .7, .2, .1)

    @staticmethod
    def get_build_set() -> ImageDataset:
        images = np.zeros([10, 16, 16, 3])
        labels = np.zeros([10, 1])

        return ImageDataset(images, labels, .7, .2, .1)


    @staticmethod
    def get_cifar10_reduced() -> ImageDataset:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


        images = np.concatenate((train_images, test_images))
        labels = np.concatenate((train_labels, test_labels))

        images = images[:4000]
        labels = labels[:4000]


        images = np.true_divide(images, 127.5)
        images = images - 1.

        return ImageDataset(images, labels, .7, .2, .1)

class ShufflerCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset: ImageDataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_begin(self, epoch, logs=None):
        self.dataset.shuffle()



class SurrogateDataset:
    def __init__(self):
        training_set = [x for x in os.listdir(res_dir) if 'surrogate_training_set' in x][-1]

        accuracies = np.loadtxt(os.path.join(res_dir, training_set, 'accuracies.csv'), delimiter=',', skiprows=1)
        embeddings = np.loadtxt(os.path.join(res_dir, training_set, 'embeddings.csv'), delimiter=',', skiprows=1)

        accuracy_cutoff = int(len(accuracies) * .75)

        accuracy_data = accuracies[:, :accuracy_cutoff]
        accuracy_labels = accuracies[:, -1:]

