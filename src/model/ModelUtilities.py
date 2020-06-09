import os
import tensorflow as tf
import numpy as np

class ModelUtilities:
    @staticmethod
    def load_keras_model(dir_path: str, model_name: str, custom_objects:dict = None):

        passed_objs = {} if custom_objects is None else custom_objects

        model = tf.keras.models.load_model(os.path.join(dir_path, model_name + '.h5'), custom_objects=passed_objs)
        return model

    @staticmethod
    def save_keras_model(keras_model, dir_path: str, model_name: str):
        # tf.keras.models.save_model(keras_model, os.path.join(dir_path, model_name + '_save'))
        # keras_model.save(os.path.join(dir_path, model_name + '_save'))
        keras_model.save(os.path.join(dir_path, model_name + '.h5'))

    @staticmethod
    def softmax(val):
        return np.exp(val) / sum(np.exp(val))