from abc import ABC, abstractmethod
import tensorflow as tf

from OperationType import OperationType


class Relu6Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, training=False, mask=None):
        return tf.nn.relu6(inputs)


class KerasOperation(ABC, tf.keras.layers.Layer):
    def __init__(self, input_dim: int, output_dim: int, stride: int, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stride = stride

    @abstractmethod
    def call(self, inputs, training=False, mask=None):
        pass

    @abstractmethod
    def rebuild_batchnorm(self):
        pass

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'stride': self.stride
        })
        return config

    @abstractmethod
    def add_self_to_parser_counts(self, parser):
        pass

    def __repr__(self):
        return f'kop     {self.stride}'


class SeperableConvolutionOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int, kernel_size: int, use_normalization: bool = True, **kwargs):
        super().__init__(input_dim, output_dim, stride, **kwargs)
        self.activation_layer = Relu6Layer()
        self.convolution_layer = tf.keras.layers.SeparableConv2D(output_dim, kernel_size, self.stride, 'same')
        self.normalization_layer = None
        self.use_normalization = use_normalization
        self.kernel_size = kernel_size

        if use_normalization:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.activation_layer(layer)
        layer = self.convolution_layer(layer)
        if self.normalization_layer is not None:
            layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'use_normalization': self.use_normalization
        })
        return config

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('relu6_layer')
        parser.get_next_name('seperable_conv2d')
        parser.get_next_name('batch_normalization')

    def __repr__(self):
        return f'sepcv {self.kernel_size} {self.stride}'


class AveragePoolingOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int, pool_size: int, **kwargs):
        super().__init__(input_dim, output_dim, stride, **kwargs)
        self.pool_size = pool_size
        self.pooling_layer = tf.keras.layers.AveragePooling2D(pool_size, strides=stride, padding='same')
        self.activation_layer = Relu6Layer()

        if self.input_dim != self.output_dim:
            self.conv_redux = tf.keras.layers.Conv2D(output_dim, 1, 1, 'same')
            self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.pooling_layer(layer)
        layer = self.activation_layer(layer)

        if self.input_dim != self.output_dim:
            layer = self.conv_redux(layer)
            layer = self.norm(layer)

        return layer

    def rebuild_batchnorm(self):
        if self.input_dim != self.output_dim and self.normalization_layer is not None:
            self.norm = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size
        })
        return config

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('relu6_layer')
        parser.get_next_name('average_pooling2d')

        if self.input_dim != self.output_dim:
            parser.get_next_name('conv2d')
            parser.get_next_name('batch_normalization')

    def __repr__(self):
        return f'avpl  {self.pool_size} {self.stride}'


class MaxPoolingOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int, pool_size: int, **kwargs):
        super().__init__(input_dim, output_dim, stride, **kwargs)
        self.pool_size = pool_size
        self.pooling_layer = tf.keras.layers.MaxPool2D(pool_size, strides=stride, padding='same')
        self.activation_layer = Relu6Layer()

        if self.input_dim != self.output_dim:
            self.conv_redux = tf.keras.layers.Conv2D(output_dim, 1, 1, 'same')
            self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.pooling_layer(layer)
        layer = self.activation_layer(layer)

        if self.input_dim != self.output_dim:
            layer = self.conv_redux(layer)
            layer = self.norm(layer)

        return layer

    def rebuild_batchnorm(self):
        if self.input_dim != self.output_dim and self.normalization_layer is not None:
            self.norm = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size
        })
        return config

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('relu6_layer')
        parser.get_next_name('max_pool2d')

        if self.input_dim != self.output_dim:
            parser.get_next_name('conv2d')
            parser.get_next_name('batch_normalization')

    def __repr__(self):
        return f'mxpl  {self.pool_size} {self.stride}'


class DoublySeperableConvoutionOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int, kernel_size: int, use_normalization: bool = True, **kwargs):
        super().__init__(input_dim, output_dim, stride, **kwargs)
        self.activation_layer = Relu6Layer()
        self.convolution_layer_1 = tf.keras.layers.SeparableConv2D(output_dim, (kernel_size, 1), (self.stride, 1), 'same')
        self.convolution_layer_2 = tf.keras.layers.SeparableConv2D(output_dim, (1, kernel_size), (1, self.stride), 'same')
        self.normalization_layer = None
        self.kernel_size = kernel_size
        if use_normalization:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.activation_layer(layer)
        layer = self.convolution_layer_1(layer)
        layer = self.convolution_layer_2(layer)
        if self.normalization_layer is not None:
            layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            self.normalization_layer = tf.keras.layers.BatchNormalization()
            # self.normalization_layer.build(self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'use_normalization': self.use_normalization
        })
        return config

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('seperable_conv2d')
        parser.get_next_name('seperable_conv2d')
        parser.get_next_name('relu6_layer')
        parser.get_next_name('batch_normalization')

    def __repr__(self):
        return f'dbscv {self.kernel_size} {self.stride}'


class FactorizedReductionOperation(KerasOperation): # TODO
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, **kwargs):
        super().__init__(output_dim, stride, **kwargs)
        self.path_1_avg = None
        self.path_1_conv = None
        self.path_2_pad = None
        self.path_2_avg = None
        self.path_concat = None

        self.single_path_conv = None
        self.activation = Relu6Layer()
        self.normalization_layer = tf.keras.layers.BatchNormalization()

        if self.stride == 1:
            self.single_path_conv = tf.keras.layers.Conv2D(output_dim, 1, self.stride, 'same')
        # else:
        #     self.path_1_avg = tf.keras.layers.AveragePooling2D(1, self.stride, padding='same')
        # TODO

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.single_path_conv(layer)
        layer = self.activation(layer)
        layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            self.normalization_layer = tf.keras.layers.BatchNormalization()
            # self.normalization_layer.build(self.output_dim)

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('conv2d')
        parser.get_next_name('relu6_layer')
        parser.get_next_name('batch_normalization')


class ConvolutionalOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, kernel_size: int = 1, **kwargs):
        super().__init__(input_dim, output_dim, stride, **kwargs)

        self.kernel_size = kernel_size
        self.reduction = tf.keras.layers.Conv2D(output_dim, self.kernel_size, self.stride, 'same')
        self.activation = Relu6Layer()
        self.normalization_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.reduction(layer)
        layer = self.activation(layer)
        layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('conv2d')
        parser.get_next_name('relu6_layer')
        parser.get_next_name('batch_normalization')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def __repr__(self):
        return f'cv    {self.kernel_size} {self.stride}'


class IdentityReductionOperation(ConvolutionalOperation):
    pass #ALIAS


class IdentityOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, **kwargs):
        # if len(kwargs) > 0:
        #     super().__init__(**kwargs)
        # else:
        #     super().__init__(0, 0)
        super().__init__(input_dim, output_dim, 1, **kwargs)

    def call(self, inputs, training=False, mask=None):
        return inputs

    def rebuild_batchnorm(self):
        return

    def add_self_to_parser_counts(self, parser):
        return

    def __repr__(self):
        return f'iop      '


class DenseOperation(KerasOperation):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 1., stride: int = 1, **kwargs):
        # if len(kwargs) > 0: #TODO: REMOVE?
        #     super().__init__(output_dim, **kwargs)
        # else:
        #     super().__init__(output_dim, 1, **kwargs)
        super().__init__(input_dim, output_dim, 1, **kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.dense_layer = tf.keras.layers.Dense(output_dim)
        self.dropout_rate = dropout_rate

    @tf.function
    def call(self, inputs, training=False, mask=None):
        layer = inputs
        if training:
            layer = self.dropout_layer(layer)
        layer = self.dense_layer(layer)

        return layer

    def rebuild_batchnorm(self):
        return

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dropout_rate': self.dropout_rate
        })
        return config

    def add_self_to_parser_counts(self, parser):
        parser.get_next_name('dropout')
        parser.get_next_name('dense')


class KerasOperationFactory:
    @staticmethod
    def get_cell_operation(op_type: int, cell_input_dim: int, cell_target_dim: int, stride: int, op_attachment_index: int):
        actual_input_dim = cell_input_dim if op_attachment_index < 2 else cell_target_dim
        actual_stride = stride if op_attachment_index < 2 else 1
        return KerasOperationFactory.get_operation(op_type, actual_input_dim, cell_target_dim, actual_stride)

    @staticmethod
    def get_operation(op_type: int, input_dim: int, output_dim: int, stride: int):
        if op_type == OperationType.SEP_3X3:
            return SeperableConvolutionOperation(input_dim, output_dim, stride, 3, True)
        elif op_type == OperationType.SEP_5X5:
            return SeperableConvolutionOperation(input_dim, output_dim, stride, 5, True)
        elif op_type == OperationType.SEP_7X7:
            return SeperableConvolutionOperation(input_dim, output_dim, stride, 7, True)
        elif op_type == OperationType.AVG_3X3:
            return AveragePoolingOperation(input_dim, output_dim, stride, 3)
        elif op_type == OperationType.AVG_5X5:
            return AveragePoolingOperation(input_dim, output_dim, stride, 5)
        elif op_type == OperationType.MAX_3X3:
            return MaxPoolingOperation(input_dim, output_dim, stride, 3)
        elif op_type == OperationType.MAX_5X5:
            return MaxPoolingOperation(input_dim, output_dim, stride, 5)
        elif op_type == OperationType.CONV_3X3:
            return ConvolutionalOperation(input_dim, output_dim, stride, 3)
        elif op_type == OperationType.SEP_1X7_7X1:
            return DoublySeperableConvoutionOperation(input_dim, output_dim, stride, 7)
        else:  # OperationType.IDENTITY and everything else
            if input_dim == output_dim and stride == 1:
                return IdentityOperation(input_dim, output_dim)
            else:
                return IdentityReductionOperation(input_dim, output_dim, stride)