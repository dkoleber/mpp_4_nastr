from __future__ import annotations
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


class OperationType(IntEnum):
    IDENTITY = 0,
    SEP_3X3 = 1,
    SEP_5X5 = 2,
    SEP_7X7 = 3,
    AVG_3X3 = 4,
    MAX_3X3 = 5,
    DIL_3X3 = 6,
    SEP_1X7_7X1 = 7


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


class OperationItem(SerialData):
    def __init__(self, attachment_index: int = 0):
        self.operation_type: OperationType = OperationType.IDENTITY
        self.attachment_index: int = attachment_index
        self.actual_attachment: int = 0

    def build_operation(self, operation_input, builder: Builder):
        return builder.get_op(self.operation_type, operation_input)

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


class Group(SerialData):
    def __init__(self):
        self.operations: List[OperationItem] = []

    def build_group(self, available_inputs, builder: Builder):
        outputs = []
        attachments = []
        for operation in self.operations:
            built_op = operation.build_operation(available_inputs[operation.actual_attachment], builder)
            outputs.append(built_op)
            attachments.append(operation.actual_attachment)
        addition = builder.add(outputs)
        return addition, attachments

    def serialize(self) -> dict:
        return {'operations': [x.serialize() for x in self.operations]}

    def deserialize(self, obj: dict) -> None:
        for op in obj['operations']:
            item = OperationItem()
            item.deserialize(op)
            self.operations.append(item)


class Block(SerialData):
    def __init__(self, num_inputs: int = 0):
        self.groups: List[Group] = []
        self.num_inputs = num_inputs

    def build_block(self, block_inputs, builder, post_reduce: bool = False):
        base_input_shape = block_inputs[0].shape[-1]
        available_inputs = [builder.identity(block_inputs[0])]
        available_inputs.extend([builder.dim_change(x, base_input_shape) for x in block_inputs[1:]])
        # available_inputs = [x for x in block_inputs]
        attachments = [False for _ in range(len(self.groups) + len(block_inputs))]
        for group_num, group in enumerate(self.groups):
            with tf.name_scope(f'group_{group_num}'):
                group_output, group_attachments = group.build_group(available_inputs, builder)
                available_inputs.append(group_output)
                for attachment in group_attachments:
                    attachments[attachment] = True
                # attachments[group.index + len(block_inputs)] = False  # this is implicit since it starts as false
        unattached = [available_inputs[x] for x in range(len(attachments)) if not attachments[x]]
        concated = builder.concat(unattached)
        if post_reduce:
            concated = builder.dim_change(concated, base_input_shape)
        return concated

    def serialize(self) -> dict:
        return {
            'groups': [x.serialize() for x in self.groups],
            'num_inputs': self.num_inputs
        }

    def deserialize(self, obj: dict) -> None:
        for group in obj['groups']:
            item = Group()
            item.deserialize(group)
            self.groups.append(item)
        self.num_inputs = obj['num_inputs']


class Model(SerialData):
    def __init__(self, hyperparameters: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.blocks: List[Block] = []  # [NORMAL_CELL, REDUCTION_CELL]
        self.hyperparameters = hyperparameters
        self.keras_model = None
        self.model_name = 'evo_' + str(time.time())  # str(time.time()).split('.')[0][4:]
        self.metrics = Metrics()
        self.fitness = 0.

    def populate_with_nasnet_blocks(self):
        groups_in_block = 5
        ops_in_group = 2
        group_inputs = 2

        def get_block():
            block = Block(group_inputs)
            block.groups = [Group() for _ in range(groups_in_block)]
            for i in range(groups_in_block):
                block.groups[i].operations = [OperationItem(i + group_inputs) for _ in range(ops_in_group)]  # +2 because 2 inputs for block, range(2) because pairwise groups
                for j in range(ops_in_group):
                    block.groups[i].operations[j].actual_attachment = min(j, group_inputs - 1)
            return block

        def randomize_block(block: Block):
            for group_ind, group in enumerate(block.groups):
                # do hidden state randomization for all but first groups
                if group_ind > 0:
                    for op in group.operations:
                        op.actual_attachment = int(np.random.random() * op.attachment_index)

                # do op randomization for all groups
                for op in group.operations:
                    op.operation_type = int(np.random.random() * OperationType.SEP_1X7_7X1)

        normal_block = get_block()
        reduction_block = get_block()

        randomize_block(normal_block)
        randomize_block(reduction_block)

        self.blocks.append(normal_block)  # normal block
        self.blocks.append(reduction_block)  # reduction block

    def build_graph(self, graph_input):
        builder = Builder()

        block_ops = []
        for block in self.blocks:
            block_ops.append(block.build_block)

        with tf.name_scope(f'initial_resize'):
            resized_input = builder.dim_change(graph_input, self.hyperparameters.parameters['INITIAL_LAYER_DIMS'])
        previous_output = resized_input
        block_input = [resized_input, previous_output]
        for layer in range(self.hyperparameters.parameters['CELL_LAYERS']):
            for normal_cells in range(self.hyperparameters.parameters['NORMAL_CELL_N']):
                with tf.name_scope(f'normal_cell_{layer}_{normal_cells}'):
                    block_output = block_ops[0](block_input, builder, self.hyperparameters.parameters['USE_POST_BLOCK_REDUCE'])
                    block_input = [block_output, previous_output]
                    previous_output = block_output
            with tf.name_scope(f'reduction_cell_{layer}'):
                block_output = block_ops[1](block_input, builder, self.hyperparameters.parameters['USE_POST_BLOCK_REDUCE'])
            if layer != self.hyperparameters.parameters['CELL_LAYERS'] - 1:
                # don't add a reduction layer at the very end of the graph before the fully connected layer
                with tf.name_scope(f'reduction_layer_{layer}'):
                    block_output = builder.downsize(block_output)
                    previous_output = builder.downsize(previous_output)
                    block_input = [block_output, previous_output]
                    previous_output = block_output

        with tf.name_scope(f'end_block'):
            output = builder.concat(block_input)
            output = builder.flatten(output)
            output = builder.dense(output, size=64, activation=tf.nn.relu)
            output = builder.dense(output, size=10)

        return output

    def mutate(self):
        other_mutation_threshold = ((1 - self.hyperparameters.parameters['IDENTITY_THRESHOLD']) / 2.) + self.hyperparameters.parameters['IDENTITY_THRESHOLD']
        block_index = int(np.random.random() * len(self.blocks))
        select_block = self.blocks[block_index]
        group_index = int(np.random.random() * len(select_block.groups))
        select_group = select_block.groups[group_index]
        item_index = int(np.random.random() * len(select_group.operations))
        select_item = select_group.operations[item_index]
        select_mutation = np.random.random()
        mutation_string = f'mutating block {block_index}, group {group_index}, item {item_index}: '
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
                    new_attachment = int(np.random.random() * select_item.attachment_index)
                select_item.actual_attachment = new_attachment
                print(mutation_string + f'hidden state mutation from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(mutation_string + f'skipping state mutation for group 0')

        else:
            # operation mutation
            previous_op = select_item.operation_type
            select_item.operation_type = int(np.random.random() * (OperationType.SEP_1X7_7X1 + 1))
            print(mutation_string + f'operation type mutation from {previous_op} to {select_item.operation_type}')

    def serialize(self) -> dict:
        return {
            'blocks': [x.serialize() for x in self.blocks],
            'metrics': self.metrics.serialize(),
            'hyperparameters': self.hyperparameters.serialize()
        }

    def deserialize(self, obj: dict) -> None:
        for block in obj['blocks']:
            item = Block()
            item.deserialize(block)
            self.blocks.append(item)

        self.metrics = Metrics()
        self.metrics.deserialize(obj['metrics'])

        self.hyperparameters = Hyperparameters()
        self.hyperparameters.deserialize(obj['hyperparameters'])

    def evaluate(self, dataset: Dataset) -> None:
        # keras_graph = tfp.keras.backend.get_session().graph
        keras_graph = tf.Graph()

        with keras_graph.as_default():

            if self.keras_model is None:
                build_time = time.time()
                model_input = tf.keras.Input(shape=dataset.images_shape)
                model_output = self.build_graph(model_input)
                self.keras_model = tf.keras.Model(inputs=model_input, outputs=model_output)
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
        os.mkdir(dir_name)
        _write_model_structure_to_json(self, dir_name, self.model_name)

    def save_graph(self, dir_path: str = model_save_dir):
        dir_name = os.path.join(dir_path, self.model_name)
        if self.keras_model is not None:
            _save_keras_model(self.keras_model, dir_name, self.model_name)

    def clear_graph(self):
        self.keras_model = None

    def duplicate(self) -> Model:
        return copy.deepcopy(self)

    def load_graph(self, dir_path: str = model_save_dir) -> bool:
        dir_name = os.path.join(dir_path, self.model_name)

        contained_files = os.listdir(dir_name)
        contains_keras_model = False

        for fl in contained_files:
            if len(fl) > 3 and fl[-3:] == '.h5':
                contains_keras_model = True

        if contains_keras_model:
            self.keras_model = _load_keras_model(dir_name, self.model_name)
            return True
        else:
            return False

    @staticmethod
    def load(dir_path: str, name: str, load_graph: bool = False) -> Model:
        dir_name = os.path.join(dir_path, name)
        if not os.path.exists(dir_name):
            print('Model does not exist at specified location')
            return Model()

        result = _load_model_structure_from_json(dir_name, name)
        result.model_name = name

        result.load_graph(dir_path)

        return result

    @staticmethod
    def load_most_recent_model() -> Model:
        models = os.listdir(model_save_dir)
        if len(models) == 0:
            print('No saved models exist to load')
            return Model()

        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # sort models by a number attached to the end of their name
        name = models[-1]
        return Model.load(model_save_dir, name)


def _write_model_structure_to_json(model: Model, dir_path: str, name: str) -> None:
    serialized = model.serialize()
    write_json_to_file(serialized, dir_path, name)


def _load_model_structure_from_json(dir_path: str, name: str) -> Model:
    serialized = read_json_from_file(dir_path, name)
    model = Model()
    model.deserialize(serialized)
    return model


def _load_keras_model(dir_path: str, name: str):
    model = load_model(os.path.join(dir_path, name + '.h5'))
    return model


def _save_keras_model(keras_model, dir_path: str, model_name: str):
    keras_model.save(os.path.join(dir_path, model_name + '.h5'))


if __name__ == '__main__':
    pass
