import tensorflow as tf
from Dataset import Dataset
import os
from FileManagement import *
import time

class SwapModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.const_layer_1 = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='const1')

        self.swap_layer_2 = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='swap2')

        self.const_layer_3 = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='const3')

        self.flatten_layer_4 = tf.keras.layers.Flatten()

        self.dense_5 = tf.keras.layers.Dense(64)

        self.dense_6 = tf.keras.layers.Dense(10)


    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.const_layer_1(layer)
        layer = self.swap_layer_2(layer)
        layer = self.const_layer_3(layer)
        layer = self.flatten_layer_4(layer)
        layer = self.dense_5(layer)
        layer = self.dense_6(layer)
        return layer

    def swap(self):
        new_size = 128
        self.swap_layer_2 = tf.keras.layers.SeparableConv2D(128, 3, 1, 'same', activation='relu', name='swapped2')
        self.swap_layer_2.build([None, None, None, 64])
        self.const_layer_3.build([None, None, None, 128])


def test1():
    dataset = Dataset.get_cifar10()

    model = SwapModel()


    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(dataset.train_images, dataset.train_labels, epochs=1)

    model.save('test_save')

    model.evaluate(dataset.test_images, dataset.test_labels)

    model.summary()
    model.swap()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()
    print(model.layers)

    model = tf.keras.models.load_model('test_save')
    model.summary()

    model.evaluate(dataset.test_images, dataset.test_labels)
    model.fit(dataset.train_images, dataset.train_labels, epochs=1)
    model.evaluate(dataset.test_images, dataset.test_labels)



class ModelHolder:
    def __init__(self):
        with tf.name_scope('conv_layers'):
            self.lay1 = tf.keras.layers.Conv2D(4, 1, 1, 'same', activation='relu', name='orig1')
            self.lay2 = tf.keras.layers.Conv2D(4, 1, 1, 'same', activation='relu', name='orig2')
            self.lay3 = tf.keras.layers.Conv2D(4, 1, 1, 'same', activation='relu', name='orig3')
        self.dense = tf.keras.layers.Dense(10, name='dense')
    def swap(self):

        new_size = 8

        old_shape = self.lay2.get_input_shape_at(0)
        new_shape = [x for x in old_shape]
        new_shape[-1] = new_size


        self.lay2 = tf.keras.layers.Conv2D(new_size, 3, 1, 'same', activation='relu', name='swapped2')
        self.lay2.build(old_shape)
        self.lay3.build(new_shape)


    def build(self, model_input):
        with tf.name_scope('conv_layers'):
            layer = self.lay1(model_input)
            layer = self.lay2(layer)
            layer = self.lay3(layer)
        layer = tf.keras.layers.Flatten()(layer)
        layer = self.dense(layer)
        keras_model = tf.keras.Model(inputs=model_input, outputs=layer)
        return keras_model

def test2():
    dataset = Dataset.get_build_set()

    model_holder = ModelHolder()

    model_input = tf.keras.Input([16, 16, 3])
    keras_model = model_holder.build(model_input)


    print(model_holder.lay2.get_weights())

    model_holder.swap()
    keras_model = model_holder.build(model_input)

    print(model_holder.lay2.get_weights())


    # model_output = keras_model(model_input)
    # temp_model = tf.keras.Model(inputs=model_input, outputs=model_output)

    keras_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))


    # logdir = os.path.join(tensorboard_dir, 'sandbox_test_' + str(time.time()))
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=False)


    keras_model.fit(dataset.train_images, dataset.train_labels, epochs=1)

    tf.keras.utils.plot_model(keras_model, 'model_image.png', expand_nested=True, show_layer_names=True, show_shapes=True)

    # with writer.as_default():
    #     tf.summary.trace_export(name='sandbox_model_trace', step=0, profiler_outdir=logdir)


def test3():
    dataset = Dataset.get_build_set()

    keras_model = SwapModel()

    keras_model._is_graph_network = True

    # model_output = keras_model(model_input)
    # temp_model = tf.keras.Model(inputs=model_input, outputs=model_output)

    keras_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # logdir = os.path.join(tensorboard_dir, 'sandbox_test_' + str(time.time()))
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=False)

    keras_model.fit(dataset.train_images, dataset.train_labels, epochs=1)


    tf.keras.utils.plot_model(keras_model, 'model_image.png', expand_nested=True, show_layer_names=True, show_shapes=True)

    # with writer.as_default():
    #     tf.summary.trace_export(name='sandbox_model_trace', step=0, profiler_outdir=logdir)

if __name__ == '__main__':
    test2()