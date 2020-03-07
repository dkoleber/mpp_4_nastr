from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import tensorflow.python as tfp
import tensorflow as tf
from typing import List
from enum import IntEnum
import time
from tensorflow.keras.models import load_model
from Dataset import Dataset
import copy
from FileManagement import *
from Metrics import Metrics
from OperationType import OperationType
from SGDR import SGDR
from SerialData import SerialData
from Hyperparameters import Hyperparameters


HERE = os.path.dirname(os.path.abspath(__file__))

'''
TODO:

=HIGH PRIORITY=
- soft vs hard fitness curve for accuracy / time tradeoff
- config determines fitness calculator


=MEDIUM PRIORITY=
- add researched selection routine
- non-overlap selection for aging selection ***********
- increasing epochs over time?

=LOW PRIORITY=
- eager model surgery ?

=FOR CONSIDERATION=
- should block output be dim reduced after concat?

=EXPERIMENTS=
coorelation between training accuracy vs late training accuracy
coorelation between FLOPS vs size vs time
coorelation betweeen accuracy of low filter numbers vs high

'''

'''

model surgery
incresaing epochs
pareto sampling
scheduled droppath
cosine annealing










1. Aging Selection
    1. select a set S candidates from population P
    2. select a set N candidates from S to have children (set M)
    3. mutate all candidates in M
    4. train all children in M
    5. Add all children in M to P
    6. Remove the |M| oldest candidates from P 

2. Tournament Selection
    select two individuals, kill the less fit one, and the more fit one has a child
'''


tf.compat.v1.disable_eager_execution()



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

    def _get_op(self, op_type: OperationType, output_shape):
        if  OperationType.SEP_3X3 <= op_type <= OperationType.SEP_1X7_7X1:
            layer = tf.keras.layers.Conv2D(output_shape, 3, 1, 'same', activation=self.activation_function)
        else:  # OperationType.IDENTITY and everything else
            layer = self.dim_change(output_shape)
        return layer

    def add(self):
        return tf.keras.layers.Add()

    def concat(self):
        return tf.keras.layers.Concatenate(axis=3)

    def dense(self, size: int = 10, activation=None):
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

    def batch_norm(self):
        return tf.keras.layers.BatchNormalization()

    def layer_norm(self):
        return tf.keras.layers.LayerNormalization(axis=3)


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
        self.model_data: ModelDataHolder = None
        self.keras_graph: tf.Graph = None
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.keras_session = None

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


                if self.model_data is not None:
                    with self.keras_graph.as_default():
                        self.model_data.hidden_state_mutation(self.hyperparameters, cell_index, group_index, item_index, new_attachment, select_item.operation_type)
                select_item.actual_attachment = new_attachment
                print(mutation_string + f'hidden state mutation from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(mutation_string + f'skipping state mutation for group 0')

        else:
            # operation mutation
            previous_op = select_item.operation_type
            select_item.operation_type = int(np.random.random() * (OperationType.SEP_1X7_7X1 + 1))
            if previous_op != select_item.operation_type and self.model_data is not None:
                with self.keras_graph.as_default():
                    self.model_data.operation_mutation(self.hyperparameters, cell_index, group_index, item_index, select_item.operation_type)
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
        if self.keras_graph is None:
            self.keras_graph = tf.Graph()

        self.optimizer = tf.keras.optimizers.Adam(self.hyperparameters.parameters['LEARNING_RATE'])

        with self.keras_graph.as_default() as graph:

            build_time = 0.

            if self.keras_model is None:
                build_time = time.time()
                self.model_data = ModelDataHolder(self)
                model_input = tf.keras.Input(input_shape)
                self.keras_model = self.model_data.build(model_input)
                build_time = time.time() - build_time

            compile_time = time.time()
            self.keras_model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            compile_time = time.time() - compile_time

            self.metrics.metrics['build_time'] = build_time
            self.metrics.metrics['compile_time'] = compile_time

    def evaluate(self, dataset: Dataset) -> None:
        with self.keras_graph.as_default():
            train_time = time.time()

            BATCH_SIZE = 64
            sgdr = SGDR(0.01, 0.001, BATCH_SIZE, len(dataset.train_labels), .25)

            self.keras_model.fit(dataset.train_images, dataset.train_labels, batch_size=BATCH_SIZE, epochs=self.hyperparameters.parameters['TRAIN_EPOCHS'], callbacks=[sgdr])
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

    def plot_graph(self, dir_path):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        had_keras_model: bool = self.keras_model is not None
        if not had_keras_model:
            self.build_model([16, 16, 3])

        tf.keras.utils.plot_model(self.keras_model, os.path.join(dir_name, self.model_name + '.png'), expand_nested=True, show_layer_names=False, show_shapes=True)

        if not had_keras_model:
            self.clear_graph()

    def save_graph(self, dir_path: str = model_save_dir):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        save_time = time.time()
        with self.keras_graph.as_default():
            if self.keras_model is not None:
                _save_keras_model(self.keras_model, dir_name, self.model_name)
        save_time = time.time() - save_time
        self.metrics.metrics['save_time'] = save_time

    def clear_graph(self):
        self.keras_model = None
        self.keras_graph = None
        self.model_data = None
        self.optimizer = None

    def produce_child(self) -> MetaModel:
        result: MetaModel = MetaModel(self.hyperparameters)
        result.cells = copy.deepcopy(self.cells)

        result.optimizer = self.optimizer
        result.keras_graph = self.keras_graph
        result.keras_model = self.keras_model
        result.model_data = self.model_data

        self.optimizer = None
        self.keras_graph = None
        self.keras_model = None
        self.model_data = None

        return result

    def load_graph(self, dir_path: str = model_save_dir) -> bool:
        dir_name = os.path.join(dir_path, self.model_name)

        contained_files = os.listdir(dir_name)
        contains_keras_model = False

        for fl in contained_files:
            if len(fl) > 3 and fl[-3:] == '.pb':
                contains_keras_model = True

        if contains_keras_model:
            self.keras_graph = tf.Graph()
            with self.keras_graph.as_default():
                self.keras_model = _load_keras_model(dir_name, self.model_name)
            return True
        else:
            return False

    @staticmethod
    def load(dir_path: str, name: str, load_graph: bool = False) -> MetaModel:
        dir_name = os.path.join(dir_path, name)
        if not os.path.exists(dir_name):
            print('Model does not exist at specified location')
            return MetaModel()

        serial_data = _load_serial_data_from_json(dir_name, name)
        result = MetaModel()
        result.deserialize(serial_data)

        result.load_graph(dir_path)

        return result


class OperationDataHolder:
    def __init__(self, op_type: int, size: int):
        builder = KerasBuilder()
        self.size = size
        self.ops = builder.get_op(op_type, size)
    def build(self, inputs):
        layer = inputs
        for op in self.ops:
            layer = op(layer)
        return layer
    def compile(self, input_shape):
        previous_shape = input_shape
        for op in self.ops:
            op.build(previous_shape)
            previous_shape = op.get_output_shape_at(0)



class GroupDataHolder:
    def __init__(self, size: int, meta_group: MetaGroup):
        builder = KerasBuilder()
        self.ops: List[tf.keras.layers.Layer] = []
        self.attachments: List[int] = []

        for op in meta_group.operations:
            self.ops.append(OperationDataHolder(op.operation_type, size))
            self.attachments.append(op.actual_attachment)

        self.addition_layer = builder.add()

    def build(self, inputs):
        results = []
        for index, obj in enumerate(self.ops):
            results.append(obj.build(inputs[self.attachments[index]]))

        return self.addition_layer(results)


class CellDataHolder:
    def __init__(self, size: int, meta_cell: MetaCell, use_post_reduce:bool = True):
        builder = KerasBuilder()

        self.groups: List[GroupDataHolder] = []

        used_group_indexes: List[int] = [0, 1]  # last cell, cell before that
        for group in meta_cell.groups:
            self.groups.append(GroupDataHolder(size, group))
            for op in group.operations:
                used_group_indexes.append(op.actual_attachment)

        self.unused_group_indexes: List[int] = [x for x in range(len(meta_cell.groups) + 2) if x not in used_group_indexes]

        self.post_concat = builder.concat()
        self.post_reduce = None
        if use_post_reduce:
            self.post_reduce = builder.dim_change(size)
        self.post_identity = builder.identity()

    def build(self, inputs):
        available_inputs = [x for x in inputs]
        for group in self.groups:
            group_output = group.build(available_inputs)
            available_inputs.append(group_output)

        result = available_inputs[-1]

        if len(self.unused_group_indexes) > 1:
            concat_groups = [available_inputs[x] for x in self.unused_group_indexes]
            result = self.post_concat(concat_groups)

        if self.post_reduce is not None:
            result = self.post_reduce(result)

        results = self.post_identity([result, inputs[0]])
        # results = [result, inputs[0]]
        return results


class ReductionCellDataHolder(CellDataHolder):
    def __init__(self, size: int, meta_cell: MetaCell, use_post_reduce:bool = True, expansion_factor: int = 1):
        super().__init__(size, meta_cell, use_post_reduce)

        builder = KerasBuilder()

        self.post_reduce_current = builder.downsize(int(size * expansion_factor))
        self.post_reduce_previous = builder.downsize(int(size * expansion_factor))

    def build(self, inputs):
        [current_result, previous_cell_result] = super().build(inputs)
        downsize_current = self.post_reduce_current(current_result)
        downsize_previous = self.post_reduce_previous(previous_cell_result)
        return [downsize_current, downsize_previous]


class ModelDataHolder:
    def __init__(self, meta_model:MetaModel):

        builder = KerasBuilder()

        model_size = meta_model.hyperparameters.parameters['INITIAL_LAYER_DIMS']
        use_post_reduce = meta_model.hyperparameters.parameters['USE_POST_BLOCK_REDUCE']

        self.cells: List[CellDataHolder] = []

        # create blocks based on meta model
        for layer in range(meta_model.hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(meta_model.hyperparameters.parameters['NORMAL_CELL_N']):
                self.cells.append(CellDataHolder(model_size, meta_model.cells[0], use_post_reduce))
            if layer != meta_model.hyperparameters.parameters['CELL_LAYERS'] - 1:
                self.cells.append(ReductionCellDataHolder(model_size, meta_model.cells[1], use_post_reduce, meta_model.hyperparameters.parameters['LAYER_EXPANSION_FACTOR']))
            model_size = int(model_size * meta_model.hyperparameters.parameters['LAYER_EXPANSION_FACTOR'])

        with tf.name_scope('end_block'):
            self.initial_resize = builder.dim_change(meta_model.hyperparameters.parameters['INITIAL_LAYER_DIMS'])
            self.final_concat = builder.concat()
            self.final_flatten = builder.flatten()
            # self.final_dense_1 = builder.dense(size=64, activation=tf.nn.relu)
            self.final_dense_2 = builder.dense(size=10)

    def operation_mutation(self, hyperparameters:Hyperparameters, cell_index: int, group_index: int, operation_index: int, new_operation: int):
        actual_cell_index = 0

        builder = KerasBuilder()

        def mutate_layer(index):
            previous_input_shape = self.cells[index].groups[group_index].ops[operation_index].get_input_shape_at(0)
            self.cells[index].groups[group_index].ops[operation_index] = OperationDataHolder(new_operation, previous_input_shape[-1])
            self.cells[index].groups[group_index].ops[operation_index].compile(previous_input_shape)

        # print(f'++ cell index: {cell_index}, group_index: {group_index}, operation_index:')

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

        builder = KerasBuilder()

        def mutate_layer(index):
            previous_input_shape = self.cells[index].groups[group_index].ops[operation_index].get_input_shape_at(0)
            self.cells[index].groups[group_index].ops[operation_index] = OperationDataHolder(operation_type, previous_input_shape[-1])
            self.cells[index].groups[group_index].attachments[operation_index] = new_hidden_state

        for layer in range(hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(hyperparameters.parameters['NORMAL_CELL_N']):
                if cell_index == 0:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1
            if layer != hyperparameters.parameters['CELL_LAYERS'] - 1:
                if cell_index == 1:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1

    def build(self, inputs):
        previous_output = self.initial_resize(inputs)

        previous_output = [previous_output, previous_output]

        for cell in self.cells:
            previous_output = cell.build(previous_output)

        output = self.final_concat(previous_output)
        output = self.final_flatten(output)
        output = self.final_dense_1(output)
        output = self.final_dense_2(output)

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

