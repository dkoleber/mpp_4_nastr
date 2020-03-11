from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import tensorflow.python as tfp
import tensorflow as tf
from typing import List
from enum import IntEnum
import time
from tensorflow.keras.models import load_model
from Dataset import Dataset, ShufflerCallback
import copy
from FileManagement import *
from Metrics import Metrics
from OperationType import OperationType
from SGDR import SGDR
from SerialData import SerialData
from Hyperparameters import Hyperparameters


HERE = os.path.dirname(os.path.abspath(__file__))




tf.compat.v1.disable_eager_execution()


def print_vars(vars):
    for var in vars:
        print(var)


class KerasBuilder:
    def __init__(self):
        self.activation_function = tf.nn.relu
        self.normalization_layer = None

    def get_op(self, op_type: OperationType, output_shape):
        result = []

        if op_type == OperationType.SEP_3X3:
            result.append(tf.keras.layers.SeparableConv2D(output_shape, 3, 1, 'same', activation=self.activation_function))
        elif op_type == OperationType.SEP_5X5:
            result.append(tf.keras.layers.SeparableConv2D(output_shape, 5, 1, 'same', activation=self.activation_function))
        elif op_type == OperationType.SEP_7X7:
            result.append(tf.keras.layers.SeparableConv2D(output_shape, 7, 1, 'same', activation=self.activation_function))
        elif op_type == OperationType.AVG_3X3:
            result.append(tf.keras.layers.AveragePooling2D(3, 1, 'same'))
        elif op_type == OperationType.MAX_3X3:
            result.append(tf.keras.layers.MaxPool2D(3, 1, 'same'))
        elif op_type == OperationType.DIL_3X3:
            result.append(tf.keras.layers.Conv2D(output_shape, 3, 1, 'same', dilation_rate=2, activation=self.activation_function))
        elif op_type == OperationType.SEP_1X7_7X1:
            result.append(tf.keras.layers.Conv2D(output_shape, (1, 7), 1, 'same', activation=self.activation_function))
            result.append(tf.keras.layers.Conv2D(output_shape, (7, 1), 1, 'same', activation=self.activation_function))
        else:  # OperationType.IDENTITY and everything else
            result.append(self.dim_change(output_shape))

        return result

    def add(self):
        return tf.keras.layers.Add()

    def concat(self):
        return tf.keras.layers.Concatenate(axis=3)

    def dense(self, size, activation=None):
        layer = tf.keras.layers.Dense(size, activation=activation)
        return layer

    def flatten(self):
        return tf.keras.layers.Flatten()

    def identity(self):
        return tf.keras.layers.Lambda(lambda x: x)

    def dim_change(self, output_shape):
        layer = tf.keras.layers.Conv2D(output_shape, 1, 1, 'same', activation=self.activation_function)
        return layer

    def downsize(self, output_shape):
        layer = tf.keras.layers.Conv2D(output_shape, 3, 2, 'same', activation=self.activation_function)

        return layer



class MetaOperation(SerialData):
    def __init__(self, attachment_index: int = 0):
        self.operation_type: OperationType = OperationType.IDENTITY
        self.attachment_index: int = attachment_index
        self.actual_attachment: int = 0

    def serialize(self) -> dict:
        return {
            'operation_type': self.operation_type,
            'attachment_index': self.attachment_index,
            'actual_attachment': self.actual_attachment
        }

    def deserialize(self, obj: dict) -> None:
        self.operation_type = obj['operation_type']
        self.attachment_index = obj['attachment_index']
        self.actual_attachment = obj['actual_attachment']


class MetaGroup(SerialData):
    def __init__(self):
        self.operations: List[MetaOperation] = []

    def serialize(self) -> dict:
        return {'operations': [x.serialize() for x in self.operations]}

    def deserialize(self, obj: dict) -> None:
        for op in obj['operations']:
            item = MetaOperation()
            item.deserialize(op)
            self.operations.append(item)


class MetaCell(SerialData):
    def __init__(self, num_inputs: int = 0):
        self.groups: List[MetaGroup] = []

    def serialize(self) -> dict:
        return {
            'groups': [x.serialize() for x in self.groups],
        }

    def deserialize(self, obj: dict) -> None:
        for group in obj['groups']:
            item = MetaGroup()
            item.deserialize(group)
            self.groups.append(item)


class MetaModel(SerialData):
    def __init__(self, hyperparameters: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.cells: List[MetaCell] = []  # [NORMAL_CELL, REDUCTION_CELL]
        self.hyperparameters = hyperparameters

        self.model_name = 'evo_' + str(time.time())
        self.metrics = Metrics()
        self.fitness = 0.

        self.keras_model: tf.keras.Model = None
        self.keras_model_data: ModelDataHolder = None

    def container_name(self):
        return self.model_name + '_container'

    def populate_with_nasnet_metacells(self):
        groups_in_block = 5
        ops_in_group = 2
        group_inputs = 2

        def get_cell():
            cell = MetaCell(group_inputs)
            cell.groups = [MetaGroup() for _ in range(groups_in_block)]
            for i in range(groups_in_block):
                cell.groups[i].operations = [MetaOperation(i + group_inputs) for _ in range(ops_in_group)]  # +2 because 2 inputs for cell, range(2) because pairwise groups
                for j in range(ops_in_group):
                    cell.groups[i].operations[j].actual_attachment = min(j, group_inputs - 1)
            return cell

        def randomize_cell(cell: MetaCell):
            for group_ind, group in enumerate(cell.groups):
                # do hidden state randomization for all but first groups
                if group_ind > 0:
                    for op in group.operations:
                        op.actual_attachment = int(np.random.random() * op.attachment_index)

                # do op randomization for all groups
                for op in group.operations:
                    op.operation_type = int(np.random.random() * OperationType.SEP_1X7_7X1)

        normal_cell = get_cell()
        reduction_cell = get_cell()

        randomize_cell(normal_cell)
        randomize_cell(reduction_cell)

        self.cells.append(normal_cell)
        self.cells.append(reduction_cell)

    def mutate(self):
        other_mutation_threshold = ((1. - self.hyperparameters.parameters['IDENTITY_THRESHOLD']) / 2.) + self.hyperparameters.parameters['IDENTITY_THRESHOLD']
        cell_index = int(np.random.random() * len(self.cells))
        select_block = self.cells[cell_index]
        group_index = int(np.random.random() * len(select_block.groups))
        select_group = select_block.groups[group_index]
        item_index = int(np.random.random() * len(select_group.operations))
        select_item = select_group.operations[item_index]
        select_mutation = np.random.random()
        mutation_string = f'mutating cell {cell_index}, group {group_index}, item {item_index}: '
        if select_mutation < self.hyperparameters.parameters['IDENTITY_THRESHOLD']:
            # identity mutation
            print(mutation_string + 'identity mutation')
        elif self.hyperparameters.parameters['IDENTITY_THRESHOLD'] < select_mutation < other_mutation_threshold:
            # hidden state mutation = change inputs

            # don't try to change the state of the first group since it need to point to the first two inputs of the block
            if group_index != 0:
                previous_attachment = select_item.actual_attachment
                new_attachment = previous_attachment
                # ensure that the mutation doesn't result in the same attachment as before
                while new_attachment == previous_attachment:
                    new_attachment = int(np.random.random() * select_item.attachment_index) #TODO: EXCLUSIVE RANDOM

                if self.keras_model_data is not None:
                    self.keras_model_data.hidden_state_mutation(self.hyperparameters, cell_index, group_index, item_index, new_attachment, select_item.operation_type)
                select_item.actual_attachment = new_attachment
                print(mutation_string + f'hidden state mutation from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(mutation_string + f'skipping state mutation for group 0')

        else:
            # operation mutation
            previous_op = select_item.operation_type
            select_item.operation_type = int(np.random.random() * (OperationType.SEP_1X7_7X1 + 1))
            if previous_op != select_item.operation_type and self.keras_model_data is not None:
                self.keras_model_data.operation_mutation(self.hyperparameters, cell_index, group_index, item_index, select_item.operation_type)
            print(mutation_string + f'operation type mutation from {previous_op} to {select_item.operation_type}')

    def serialize(self) -> dict:
        return {
            'blocks': [x.serialize() for x in self.cells],
            'metrics': self.metrics.serialize(),
            'hyperparameters': self.hyperparameters.serialize(),
            'model_name': self.model_name
        }

    def deserialize(self, obj: dict) -> None:
        for block in obj['blocks']:
            item = MetaCell()
            item.deserialize(block)
            self.cells.append(item)
        self.model_name = obj['model_name']
        self.metrics = Metrics()
        self.metrics.deserialize(obj['metrics'])
        self.hyperparameters = Hyperparameters()
        self.hyperparameters.deserialize(obj['hyperparameters'])

    def build_model(self, input_shape) -> None:
        if self.keras_model is None:
            print('creating new model')
            build_time = time.time()
            self.keras_model_data = ModelDataHolder(self)
            model_input = tf.keras.Input(input_shape)
            self.keras_model = self.keras_model_data.build(model_input)
            build_time = time.time() - build_time
            optimizer = tf.keras.optimizers.Adam(self.hyperparameters.parameters['LEARNING_RATE'])

            compile_time = time.time()
            self.keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            compile_time = time.time() - compile_time

            self.metrics.metrics['build_time'] = build_time
            self.metrics.metrics['compile_time'] = compile_time
        else:
            print('reusing previous keras model')

    def evaluate(self, dataset: Dataset) -> None:
        BATCH_SIZE = 64
        sgdr = SGDR(0.01, 0.001, BATCH_SIZE, len(dataset.train_labels), .25)
        shuffler = ShufflerCallback(dataset)

        train_time = time.time()
        self.keras_model.fit(dataset.train_images, dataset.train_labels, batch_size=BATCH_SIZE, epochs=self.hyperparameters.parameters['TRAIN_EPOCHS'], callbacks=[sgdr, shuffler])
        train_time = time.time() - train_time

        inference_time = time.time()
        evaluated_metrics = self.keras_model.evaluate(dataset.test_images, dataset.test_labels)
        inference_time = time.time() - inference_time

        self.metrics.metrics['accuracy'] = float(evaluated_metrics[-1])
        self.metrics.metrics['average_train_time'] = train_time / float(self.hyperparameters.parameters['TRAIN_EPOCHS'] * len(dataset.train_labels))
        self.metrics.metrics['average_inference_time'] = inference_time / float(len(dataset.test_images))

    def save_metadata(self, dir_path: str = model_save_dir):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        _write_serial_data_to_json(self, dir_name, self.model_name)

    def plot_model(self, dir_path):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        had_keras_model: bool = self.keras_model is not None
        if not had_keras_model:
            self.build_model([16, 16, 3])

        tf.keras.utils.plot_model(self.keras_model, os.path.join(dir_name, self.model_name + '.png'), expand_nested=True, show_layer_names=False, show_shapes=True)

        if not had_keras_model:
            self.clear_model()

    def save_model(self, dir_path: str = model_save_dir):
        if self.keras_model is not None:
            print(f'saving graph for {self.model_name}')
            dir_name = os.path.join(dir_path, self.model_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            save_time = time.time()
            _save_keras_model(self.keras_model, dir_name, self.model_name)
            save_time = time.time() - save_time
            self.metrics.metrics['save_time'] = save_time
            print(f'finished saving graph for {self.model_name}')

    def clear_model(self):
        print(f'clearing model for {self.model_name}')
        if self.keras_model is not None:
            del self.keras_model
            self.keras_model = None
        self.keras_model_data = None
        print(f'finished clearing model for {self.model_name}')

    def produce_child(self) -> MetaModel:
        result: MetaModel = MetaModel(self.hyperparameters)
        result.cells = copy.deepcopy(self.cells)

        result.keras_model = self.keras_model
        result.keras_model_data = self.keras_model_data

        self.keras_model = None
        self.keras_model_data = None

        return result

    def load_model(self, dir_path: str = model_save_dir) -> bool:
        dir_name = os.path.join(dir_path, self.model_name)

        contained_files = os.listdir(dir_name)
        contains_keras_model = False

        for fl in contained_files:
            if len(fl) > 3 and fl[-3:] == '.pb':
                contains_keras_model = True

        if contains_keras_model:
            print(f'loading model for {self.model_name}')
            self.keras_model = _load_keras_model(dir_name, self.model_name)
            print(f'finished loading model for {self.model_name}')
            return True
        else:
            return False

    @staticmethod
    def load(dir_path: str, name: str, load_graph: bool = False) -> MetaModel:
        print(f'loading model, load_graph = {load_graph}')
        dir_name = os.path.join(dir_path, name)
        if not os.path.exists(dir_name):
            print('Model does not exist at specified location')
            return MetaModel()

        serial_data = _load_serial_data_from_json(dir_name, name)
        result = MetaModel()
        result.deserialize(serial_data)
        if load_graph:
            result.load_model(dir_path)

        return result


class Relu6Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, inputs, training=False, mask=None):
        return tf.nn.relu6(inputs)


class KerasOperation(ABC, tf.keras.layers.Layer):
    def __init__(self, output_dim: int, stride: int):
        super().__init__()
        self.output_dim = output_dim
        self.stride = stride
    @abstractmethod
    def call(self, inputs, training=False, mask=None):
        pass
    @abstractmethod
    def rebuild_batchnorm(self):
        pass


class SeperableConvolutionOperation(KerasOperation):
    def __init__(self, output_dim: int, stride: int, kernel_size: int, use_normalization: bool = True, dilation_rate: int = 1):
        super().__init__(output_dim, stride)
        self.activation_layer = Relu6Layer()
        self.convolution_layer = tf.keras.layers.SeparableConv2D(output_dim, kernel_size, self.stride, 'same', dilation_rate=dilation_rate)
        self.normalization_layer = None
        if use_normalization:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.activation_layer(layer)
        layer = self.convolution_layer(layer)
        if self.normalization_layer is not None:
            # print('using sep conv bn')
            layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            # print('building sep conv bn')
            self.normalization_layer = tf.keras.layers.BatchNormalization()
            # self.normalization_layer.build(self.output_dim)


class AveragePoolingOperation(KerasOperation):
    def __init__(self, output_dim: int, stride: int, pool_size: int):
        super().__init__(output_dim, stride)
        self.pool_size = pool_size
        self.pooling_layer = tf.keras.layers.AveragePooling2D(pool_size, strides=stride, padding='same')
        self.activation_layer = Relu6Layer()
    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.pooling_layer(layer)
        layer = self.activation_layer(layer)
        return layer
    def rebuild_batchnorm(self):
        return


class MaxPoolingOperation(KerasOperation):
    def __init__(self, output_dim: int, stride: int, pool_size: int):
        super().__init__(output_dim, stride)
        self.pool_size = pool_size
        self.pooling_layer = tf.keras.layers.MaxPool2D(pool_size, strides=stride, padding='same')
        self.activation_layer = Relu6Layer()
    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.pooling_layer(layer)
        layer = self.activation_layer(layer)
        return layer
    def rebuild_batchnorm(self):
        return


class DoublySeperableConvoutionOperation(KerasOperation):
    def __init__(self, output_dim: int, stride: int, kernel_size: int, use_normalization: bool = True):
        super().__init__(output_dim, stride)
        self.activation_layer = Relu6Layer()
        self.convolution_layer_1 = tf.keras.layers.SeparableConv2D(output_dim, (kernel_size, 1), self.stride, 'same')
        self.convolution_layer_2 = tf.keras.layers.SeparableConv2D(output_dim, (1, kernel_size), self.stride, 'same')
        self.normalization_layer = None
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


class DimensionalityReductionOperation(KerasOperation):
    def __init__(self, output_dim: int, stride: int = 1):
        super().__init__(output_dim, stride)
        self.path_1_avg = None
        self.path_1_conv = None
        self.path_2_pad = None
        self.path_2_avg = None
        self.path_concat = None

        self.single_path_conv = None

        self.normalization_layer = tf.keras.layers.BatchNormalization()

        if self.stride == 1:
            self.single_path_conv = tf.keras.layers.Conv2D(output_dim, 1, self.stride, 'same')
        # else:
        #     self.path_1_avg = tf.keras.layers.AveragePooling2D(1, self.stride, padding='same')
        # TODO

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        layer = self.single_path_conv(layer)
        layer = self.normalization_layer(layer)
        return layer

    def rebuild_batchnorm(self):
        if self.normalization_layer is not None:
            self.normalization_layer = tf.keras.layers.BatchNormalization()
            # self.normalization_layer.build(self.output_dim)


class IdentityOperation(KerasOperation):
    def __init__(self):
        super().__init__(0, 0)
        self.identity_layer = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs, training=False, mask=None):
        return self.identity_layer(inputs)

    def rebuild_batchnorm(self):
        return


class DenseOperation(KerasOperation):
    def __init__(self, output_dim: int, dropout_rate: float = 1):
        super().__init__(output_dim, 1)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.dense_layer = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False, mask=None):
        layer = inputs
        if training:
            layer = self.dropout_layer(layer)
        layer = self.dense_layer(layer)

        return layer

    def rebuild_batchnorm(self):
        return


class KerasOperationFactory:
    @staticmethod
    def get_operation(op_type: int, output_dim: int, stride: int = 1):
        if op_type == OperationType.SEP_3X3:
            return SeperableConvolutionOperation(output_dim, stride, 3)
        elif op_type == OperationType.SEP_5X5:
            return SeperableConvolutionOperation(output_dim, stride, 5)
        elif op_type == OperationType.SEP_7X7:
            return SeperableConvolutionOperation(output_dim, stride, 7)
        elif op_type == OperationType.AVG_3X3:
            return AveragePoolingOperation(output_dim, stride, 3)
        elif op_type == OperationType.MAX_3X3:
            return MaxPoolingOperation(output_dim, stride, 3)
        elif op_type == OperationType.DIL_3X3:
            return SeperableConvolutionOperation(output_dim, stride, 3, dilation_rate=2)
        elif op_type == OperationType.SEP_1X7_7X1:
            return DoublySeperableConvoutionOperation(output_dim, stride, 7)
        else:  # OperationType.IDENTITY and everything else
            return IdentityOperation()


class GroupDataHolder:
    def __init__(self, size: int, meta_group: MetaGroup):

        self.ops = []
        self.attachments: List[int] = []

        for op in meta_group.operations:
            self.ops.append(KerasOperationFactory.get_operation(op.operation_type, size))
            self.attachments.append(op.actual_attachment)

        self.addition_layer = tf.keras.layers.Add()

    def build(self, inputs):
        results = []
        for index, obj in enumerate(self.ops):
            results.append(obj(inputs[self.attachments[index]]))

        return self.addition_layer(results)


class CellDataHolder:
    def __init__(self, input_dim: int, meta_cell: MetaCell):

        self.groups: List[GroupDataHolder] = []

        used_group_indexes: List[int] = [0, 1]  # last cell, cell before that
        for group in meta_cell.groups:
            self.groups.append(GroupDataHolder(input_dim, group))
            for op in group.operations:
                used_group_indexes.append(op.actual_attachment)

        self.unused_group_indexes: List[int] = [x for x in range(len(meta_cell.groups) + 2) if x not in used_group_indexes]
        # self.output_size = input_dim * len(self.unused_group_indexes)
        self.output_size = input_dim

        self.post_concat = tf.keras.layers.Concatenate(axis=3)
        self.post_reduce = DimensionalityReductionOperation(input_dim)


    def build(self, inputs):
        available_inputs = [x for x in inputs]
        for group in self.groups:
            group_output = group.build(available_inputs)
            available_inputs.append(group_output)

        result = available_inputs[-1]

        if len(self.unused_group_indexes) > 1:
            concat_groups = [available_inputs[x] for x in self.unused_group_indexes]
            result = self.post_concat(concat_groups)

        # other_reduce = self.post_reduce(inputs[0])
        other_reduce = inputs[0]
        result = self.post_reduce(result)

        # results = self.post_identity([result, other_reduce])
        results = [result, other_reduce]
        return results


class ReductionCellDataHolder(CellDataHolder):
    def __init__(self, input_dim: int, meta_cell: MetaCell):
        super().__init__(input_dim, meta_cell)

        self.post_reduce_current = tf.keras.layers.Conv2D(self.output_size, 3, 2, 'same')
        self.post_reduce_previous = tf.keras.layers.Conv2D(self.output_size, 3, 2, 'same')

    def build(self, inputs):
        [current_result, previous_cell_result] = super().build(inputs)
        downsize_current = self.post_reduce_current(current_result)
        downsize_previous = self.post_reduce_previous(previous_cell_result)
        return [downsize_current, downsize_previous]


class ModelDataHolder:
    def __init__(self, meta_model:MetaModel):

        builder = KerasBuilder()

        initial_size = meta_model.hyperparameters.parameters['INITIAL_LAYER_DIMS']

        self.cells: List[CellDataHolder] = []

        previous_size = initial_size
        # create blocks based on meta model
        for layer in range(meta_model.hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(meta_model.hyperparameters.parameters['NORMAL_CELL_N']):
                cell = CellDataHolder(previous_size, meta_model.cells[0])
                self.cells.append(cell)
                previous_size = cell.output_size
            if layer != meta_model.hyperparameters.parameters['CELL_LAYERS'] - 1:
                cell = ReductionCellDataHolder(previous_size, meta_model.cells[1])
                self.cells.append(cell)
                previous_size = cell.output_size

        self.initial_resize = tf.keras.layers.Conv2D(initial_size, 3, 1, 'same')
        self.initial_norm = tf.keras.layers.BatchNormalization()

        self.final_flatten = builder.flatten()
        self.final_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))
        self.final_activation = Relu6Layer()
        # self.final_dropout = tf.keras.layers.Dropout(meta_model.hyperparameters.parameters['DROPOUT_RATE'])
        self.final_dense = DenseOperation(10, .5)

    def operation_mutation(self, hyperparameters:Hyperparameters, cell_index: int, group_index: int, operation_index: int, new_operation: int):
        actual_cell_index = 0

        def mutate_layer(index):
            previous_input_shape = self.cells[index].groups[group_index].ops[operation_index].get_input_shape_at(0)
            self.cells[index].groups[group_index].ops[operation_index] = KerasOperationFactory.get_operation(new_operation, previous_input_shape[-1])
            self.cells[index].groups[group_index].ops[operation_index].build(previous_input_shape)
            print('--finished building mutated layer (operation)')

        for layer in range(hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(hyperparameters.parameters['NORMAL_CELL_N']):
                if cell_index == 0:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1
            if layer != hyperparameters.parameters['CELL_LAYERS'] - 1:
                if cell_index == 1:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1

    def hidden_state_mutation(self, hyperparameters:Hyperparameters, cell_index: int, group_index: int, operation_index: int, new_hidden_state: int, operation_type: int):
        actual_cell_index = 0

        def mutate_layer(index):
            previous_input_shape = self.cells[index].groups[group_index].ops[operation_index].get_input_shape_at(0)
            self.cells[index].groups[group_index].ops[operation_index] = KerasOperationFactory.get_operation(operation_type, previous_input_shape[-1])
            self.cells[index].groups[group_index].ops[operation_index].build(previous_input_shape)
            self.cells[index].groups[group_index].attachments[operation_index] = new_hidden_state
            print('--finished building mutated layer (state)')

        for layer in range(hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(hyperparameters.parameters['NORMAL_CELL_N']):
                if cell_index == 0:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1
            if layer != hyperparameters.parameters['CELL_LAYERS'] - 1:
                if cell_index == 1:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1

    def rebuild_batchnorm(self, hyperparameters:Hyperparameters):
        print('rebuilding bn')
        for cell in self.cells:
            for group in cell.groups:
                for op in group.ops:
                    op.rebuild_batchnorm()

    def build(self, inputs):
        previous_output = self.initial_resize(inputs)
        previous_output = self.initial_norm(previous_output)

        previous_output = [previous_output, previous_output]

        for cell in self.cells:
            previous_output = cell.build(previous_output)

        # output = self.final_flatten(previous_output[0])
        output = self.final_pool(previous_output[0])
        output = self.final_activation(output)
        output = self.final_dense(output)

        return tf.keras.Model(inputs=inputs, outputs=output)


def _write_serial_data_to_json(data: SerialData, dir_path: str, name: str) -> None:
    serialized = data.serialize()
    write_json_to_file(serialized, dir_path, name)


def _load_serial_data_from_json(dir_path: str, name: str) -> dict:
    serialized = read_json_from_file(dir_path, name)
    return serialized


def _load_keras_model(dir_path: str, name: str):
    model = load_model(os.path.join(dir_path, name + '.pb'))
    return model


def _save_keras_model(keras_model, dir_path: str, model_name: str):
    keras_model.save(os.path.join(dir_path, model_name + '.pb'))


if __name__ == '__main__':
    pass

