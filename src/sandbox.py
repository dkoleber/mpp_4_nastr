import tensorflow as tf
from Dataset import Dataset


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



if __name__ == '__main__':
    test1()