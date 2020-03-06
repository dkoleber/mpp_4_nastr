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
- non-overlap selection for aging selection
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


# tf.compat.v1.disable_eager_execution()


class Builder:
    def __init__(self):
        self.count_names = {}
        self.normalization_layer = None  # self.batch_norm
        self.activation_function = tf.nn.relu

    def get_name(self, name):
        scope = tfp.keras.backend.get_session().graph.get_name_scope()
        full_name = f'{scope}/{name}'
        if full_name in self.count_names:
            self.count_names[full_name] += 1
        else:
            self.count_names[full_name] = 0
        return f'{full_name}_{self.count_names[full_name]}'

    def get_op(self, op_type: OperationType, op_input):
        input_shape = op_input.shape[-1]
        layer = op_input
        if op_type == OperationType.SEP_3X3:
            layer = tf.keras.layers.SeparableConv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_5X5:
            layer = tf.keras.layers.SeparableConv2D(input_shape, 5, 1, 'same', name=self.get_name('SEP_5X5'), activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_7X7:
            layer = tf.keras.layers.SeparableConv2D(input_shape, 7, 1, 'same', name=self.get_name('SEP_7X7'), activation=self.activation_function)(layer)
        elif op_type == OperationType.AVG_3X3:
            layer = tf.keras.layers.AveragePooling2D(3, 1, 'same', name=self.get_name('AVG_3X3'))(layer)
        elif op_type == OperationType.MAX_3X3:
            layer = tf.keras.layers.MaxPool2D(3, 1, 'same', name=self.get_name('MAX_3X3'))(layer)
        elif op_type == OperationType.DIL_3X3:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('DIL_3X3'), dilation_rate=2, activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_1X7_7X1:
            layer = tf.keras.layers.Conv2D(input_shape, (1, 7), 1, 'same', name=self.get_name('SEP_1X7'), activation=self.activation_function)(layer)
            layer = tf.keras.layers.Conv2D(input_shape, (7, 1), 1, 'same', name=self.get_name('SEP_7X1'), activation=self.activation_function)(layer)
        else:  # OperationType.IDENTITY and everything else
            layer = self.dim_change(layer, input_shape)
        return self.post_op(layer)

    def add(self, values):
        return tf.keras.layers.Add(name=self.get_name('add'))(values)

    def concat(self, values):
        if len(values) == 1:
            return self.identity(values[0])
        else:
            return tf.keras.layers.Concatenate(axis=3, name=self.get_name('concat'))(values)

    def dense(self, layer_input, size: int = 10, activation=None):
        layer = tf.keras.layers.Dense(size, name=self.get_name('dense'), activation=activation)(layer_input)
        return layer

    def flatten(self, layer_input):
        return tf.keras.layers.Flatten(name=self.get_name('flatten'))(layer_input)

    def softmax(self, layer_input):
        return tf.keras.layers.Softmax(name=self.get_name('softmax'))(layer_input)

    def identity(self, layer_input):
        return tf.keras.layers.Lambda(lambda x: x, name=self.get_name('identity'))(layer_input)

    def dim_change(self, layer_input, output_size):
        if layer_input.shape[-1] == output_size:
            return self.identity(layer_input)
        else:
            layer = tf.keras.layers.Conv2D(output_size, 1, 1, 'same', name=self.get_name('dim_redux_1x1'), activation=self.activation_function)(layer_input)
            return self.post_op(layer)

    def downsize(self, layer_input):
        input_size = layer_input.shape[-1]
        layer = tf.keras.layers.Conv2D(input_size, 3, 2, 'same', name=self.get_name('downsize'), activation=self.activation_function)(layer_input)
        return self.post_op(layer)

    def batch_norm(self, layer_input):
        return tf.keras.layers.BatchNormalization(name=self.get_name('BN'))(layer_input)

    def layer_norm(self, layer_input):
        return tf.keras.layers.LayerNormalization(axis=3, name=self.get_name('LN'))(layer_input)

    def post_op(self, layer_input):
        layer = layer_input
        if self.normalization_layer is not None:
            layer = self.normalization_layer(layer)
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

        self.keras_model: NASNet = None
        self.keras_graph: tf.Graph = None

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
        other_mutation_threshold = ((1 - self.hyperparameters.parameters['IDENTITY_THRESHOLD']) / 2.) + self.hyperparameters.parameters['IDENTITY_THRESHOLD']
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


                if self.keras_model is not None:
                    self.keras_model.hidden_state_mutation(cell_index, group_index, item_index, new_attachment)
                select_item.actual_attachment = new_attachment



                print(mutation_string + f'hidden state mutation from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(mutation_string + f'skipping state mutation for group 0')

        else:
            # operation mutation
            previous_op = select_item.operation_type
            select_item.operation_type = int(np.random.random() * (OperationType.SEP_1X7_7X1 + 1))
            if previous_op != select_item.operation_type and self.keras_model is not None:
                self.keras_model.operation_mutation(cell_index, group_index, item_index, select_item.operation_type)
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

    def evaluate(self, dataset: Dataset) -> None:
        # keras_graph = tfp.keras.backend.get_session().graph
        self.keras_graph = tf.Graph()

        with self.keras_graph.as_default() as graph:

            if self.keras_model is None:
                build_time = time.time()
                self.keras_model = NASNet(self)
                build_time = time.time() - build_time

                tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_dir, self.model_name))

                compile_time = time.time()
                optimizer = tf.keras.optimizers.Adam(self.hyperparameters.parameters['LEARNING_RATE'])
                self.keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                compile_time = time.time() - compile_time

            train_time = time.time()
            self.keras_model.fit(dataset.train_images, dataset.train_labels, epochs=self.hyperparameters.parameters['TRAIN_EPOCHS'], callbacks=[tensorboard_callback])
            train_time = time.time() - train_time

            interence_time = time.time()
            evaluated_metrics = self.keras_model.evaluate(dataset.test_images, dataset.test_labels)
            inference_time = time.time() - interence_time

            self.metrics.metrics['accuracy'] = float(evaluated_metrics[-1])
            self.metrics.metrics['build_time'] = build_time
            self.metrics.metrics['compile_time'] = compile_time
            self.metrics.metrics['average_train_time'] = train_time / float(self.hyperparameters.parameters['TRAIN_EPOCHS'] * len(dataset.train_labels))
            self.metrics.metrics['average_inference_time'] = inference_time / float(len(dataset.test_images))

    def save_metadata(self, dir_path: str = model_save_dir):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        _write_serial_data_to_json(self, dir_name, self.model_name)

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

    def duplicate(self) -> MetaModel:
        return copy.deepcopy(self)

    def load_graph(self, dir_path: str = model_save_dir) -> bool:
        dir_name = os.path.join(dir_path, self.model_name)

        contained_files = os.listdir(dir_name)
        contains_keras_model = False

        for fl in contained_files:
            if len(fl) > 3 and fl[-3:] == '.pb':
                contains_keras_model = True

        if contains_keras_model:
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


class KerasBuilder:
    def __init__(self):
        self.activation_function = tf.nn.relu
        self.normalization_layer = None

    def get_op(self, op_type: OperationType, output_shape):
        if op_type == OperationType.SEP_3X3:
            layer = tf.keras.layers.SeparableConv2D(output_shape, 3, 1, 'same', activation=self.activation_function)
        elif op_type == OperationType.SEP_5X5:
            layer = tf.keras.layers.SeparableConv2D(output_shape, 5, 1, 'same', activation=self.activation_function)
        elif op_type == OperationType.SEP_7X7:
            layer = tf.keras.layers.SeparableConv2D(output_shape, 7, 1, 'same', activation=self.activation_function)
        elif op_type == OperationType.AVG_3X3:
            layer = tf.keras.layers.AveragePooling2D(3, 1, 'same')
        elif op_type == OperationType.MAX_3X3:
            layer = tf.keras.layers.MaxPool2D(3, 1, 'same')
        elif op_type == OperationType.DIL_3X3:
            layer = tf.keras.layers.Conv2D(output_shape, 3, 1, 'same', dilation_rate=2, activation=self.activation_function)
        elif op_type == OperationType.SEP_1X7_7X1:
            layer = tf.keras.layers.Conv2D(output_shape, (1, 7), 1, 'same', activation=self.activation_function)
            layer = tf.keras.layers.Conv2D(output_shape, (7, 1), 1, 'same', activation=self.activation_function)
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


class NASGroup(tf.keras.layers.Layer):
    def __init__(self, size: int, meta_group: MetaGroup):
        super().__init__()

        builder = KerasBuilder()

        self.ops: List[tf.keras.layers.Layer] = []
        self.attachments: List[int] = []

        for op in meta_group.operations:
            self.ops.append(builder.get_op(op.operation_type, size))
            self.attachments.append(op.actual_attachment)

        self.addition_layer = builder.add()

    def call(self, inputs, training=False, mask=None):
        results = []
        for index, obj in enumerate(self.ops):
            results.append(obj(inputs[self.attachments[index]]))

        return self.addition_layer(results)


class NASCell(tf.keras.layers.Layer):
    def __init__(self, size: int, meta_cell: MetaCell):
        super().__init__()

        builder = KerasBuilder()

        self.groups: List[NASGroup] = []

        used_group_indexes: List[int] = [0, 1]  # last cell, cell before that
        for group in meta_cell.groups:
            self.groups.append(NASGroup(size, group))
            for op in group.operations:
                used_group_indexes.append(op.actual_attachment)

        self.unused_group_indexes: List[int] = [x for x in range(len(meta_cell.groups) + 2) if x not in used_group_indexes]

        self.post_concat = builder.concat()
        self.post_reduce = builder.dim_change(size)

    def call(self, inputs, training=False, mask=None):
        available_inputs = [x for x in inputs]
        for group in self.groups:
            group_output = group(available_inputs)
            available_inputs.append(group_output)

        result = available_inputs[-1]

        if len(self.unused_group_indexes) > 1:
            concat_groups = [available_inputs[x] for x in self.unused_group_indexes]
            result = self.post_concat(concat_groups)
            result = self.post_reduce(result)
        return [result, inputs[0]]


class NormalCell(NASCell):
    pass #here for semantics only


class ReductionCell(NASCell):
    def __init__(self, size: int, meta_cell: MetaCell):
        super().__init__(size, meta_cell)

        builder = KerasBuilder()

        self.post_reduce_current = builder.downsize(size)
        self.post_reduce_previous = builder.downsize(size)

    def call(self, inputs, training=False, mask=None):
        [current_result, previous_cell_result] = super().call(inputs)
        downsize_current = self.post_reduce_current(current_result)
        downsize_previous = self.post_reduce_previous(previous_cell_result)
        return [downsize_current, downsize_previous]


class NASNet(tf.keras.Model):
    def __init__(self, meta_model: MetaModel):
        super().__init__()

        builder = KerasBuilder()

        model_size = meta_model.hyperparameters.parameters['INITIAL_LAYER_DIMS']

        self.cells: List[NASCell] = []

        #create blocks based on meta model
        for layer in range(meta_model.hyperparameters.parameters['CELL_LAYERS']):
            with tf.name_scope(f'normal_cell_{layer}'):
                for normal_cells in range(meta_model.hyperparameters.parameters['NORMAL_CELL_N']):
                    self.cells.append(NormalCell(model_size, meta_model.cells[0]))
            with tf.name_scope(f'reduction_cell_{layer}'):
                if layer != meta_model.hyperparameters.parameters['CELL_LAYERS'] - 1:
                    self.cells.append(ReductionCell(model_size, meta_model.cells[1]))

        with tf.name_scope('end_block'):
            self.initial_resize = builder.dim_change(model_size)
            self.final_concat = builder.concat()
            self.final_flatten = builder.flatten()
            self.final_dense_1 = builder.dense(size=64, activation=tf.nn.relu)
            self.final_dense_2 = builder.dense(size=10)


    def operation_mutation(self, cell_index: int, group_index: int, operation_index: int, new_operation: int):
        pass

    def hidden_state_mutation(self, cell_index: int, group_index: int, operation_index: int, new_hidden_state: int):
        pass

    def call(self, inputs, training=False, mask=None):
        previous_output = self.initial_resize(inputs)

        previous_output = [previous_output, previous_output]

        for cell in self.cells:
            previous_output = cell(previous_output)

        output = self.final_concat(previous_output)
        output = self.final_flatten(output)
        output = self.final_dense_1(output)
        output = self.final_dense_2(output)

        return output


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
    #tf.compat.v1.disable_eager_execution()



    meta_model = MetaModel()
    meta_model.populate_with_nasnet_metacells()
    keras_model = NASNet(meta_model)

    optimizer = tf.keras.optimizers.Adam(meta_model.hyperparameters.parameters['LEARNING_RATE'])
    keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    logdir = os.path.join(tensorboard_dir, 'test_' + str(time.time()))
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    dataset = Dataset.get_build_set()
    keras_model.fit(dataset.train_images, dataset.train_labels, epochs=1)
    # tf.keras.utils.plot_model(keras_model, 'model_image.png', expand_nested=True, show_layer_names=True, show_shapes=True)

    with writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=logdir)

