from __future__ import annotations
import numpy as np
import tensorflow.python as tfp
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Tuple
from enum import Enum, IntEnum
import os
import time
import json
from tensorflow.keras.models import load_model
from Dataset import Dataset
from Candidate import Candidate
import copy

HERE = os.path.dirname(os.path.abspath(__file__))


res_dir = os.path.join(HERE,'..\\res\\')
tensorboard_dir = os.path.join(HERE,'..\\tensorboard\\')
model_save_dir = os.path.join(HERE, '..\\models\\')

tf.compat.v1.disable_eager_execution()

'''
TODO:

- eager model surgery
- training routine

- selection routine
- soft vs hard curve for accuracy / time tradeoff

- should block output be dim reduced after concat?
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
    
3. Tournament with combination
    select 4 individuals, kill the least fit two, and the remaining two each have a child
    the children have some random replacement beteween eachother, and then are mtuated

---

initialization types:
full passthrough
random


---

search space

- number of blocks 
+ number of groups within blocks
+ number of items within groups
+ operation used by items (identity, 
+ initialization type of variables in operation (zeros, identity aka parent, random, xavier, etc) 




hyperconfigurations:
initialization scheme (full passthrough vs random)
initialization type (parent/zeros/random)


mutations:
add/remove group
add/remove item
change operation / change input 

'''


IDENTITY_THRESHOLD = .33
NORMAL_CELL_N = 5
CELL_LAYERS = 3

INITIAL_LAYER_DIMS = 16
USE_POST_BLOCK_REDUCE = True

tf.compat.v1.disable_eager_execution()

class SerialData(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def deserialize(self, obj:dict) -> None:
        pass


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
        self.normalization_layer = None #self.batch_norm
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
        else: # OperationType.IDENTITY and everything else
            layer = self.dim_change(layer, input_shape)
        return self.post_op(layer)

    def _get_op(self, op_type: OperationType, op_input):
        input_shape = op_input.shape[-1]
        layer = op_input
        if op_type == OperationType.SEP_3X3:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_5X5:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_7X7:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.AVG_3X3:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.MAX_3X3:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.DIL_3X3:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        elif op_type == OperationType.SEP_1X7_7X1:
            layer = tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'), activation=self.activation_function)(layer)
        else: # OperationType.IDENTITY and everything else
            layer = self.dim_change(layer, input_shape)
        return self.post_op(layer)

    def add(self, values):
        return tf.keras.layers.Add(name=self.get_name('add'))(values)

    def concat(self, values):
        if len(values) == 1:
            return self.identity(values[0])
        else:
            return tf.keras.layers.Concatenate(axis=3, name=self.get_name('concat'))(values)

    def dense(self, layer_input, size = 10, activation = None):
        layer = tf.keras.layers.Dense(size, name=self.get_name('dense'), activation=activation)(layer_input)
        return layer

    def flatten(self, layer_input, scope=''):
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

    def deserialize(self, obj:dict) -> None:
        for op in obj['operations']:
            item = OperationItem()
            item.deserialize(op)
            self.operations.append(item)


class Block(SerialData):
    def __init__(self, num_inputs: int = 0):
        self.groups: List[Group] = []
        self.num_inputs = num_inputs

    def build_block(self, block_inputs, builder):
        base_input_shape = block_inputs[0].shape[-1]
        available_inputs = [builder.identity(block_inputs[0])]
        available_inputs.extend([builder.dim_change(x, base_input_shape) for x in block_inputs[1:]])
        # available_inputs = [x for x in block_inputs]
        attachments = [False for x in range(len(self.groups) + len(block_inputs))]
        for group_num, group in enumerate(self.groups):
            with tf.name_scope(f'group_{group_num}'):
                group_output, group_attachments = group.build_group(available_inputs, builder)
                available_inputs.append(group_output)
                for attachment in group_attachments:
                    attachments[attachment] = True
                # attachments[group.index + len(block_inputs)] = False  # this is implicit since it starts as false
        unattached = [available_inputs[x] for x in range(len(attachments)) if not attachments[x]]
        concated = builder.concat(unattached)
        if USE_POST_BLOCK_REDUCE:
            concated = builder.dim_change(concated, base_input_shape)
        return concated

    def serialize(self) -> dict:
        return {
            'groups': [x.serialize() for x in self.groups],
            'num_inputs': self.num_inputs
        }

    def deserialize(self, obj:dict) -> None:
        for group in obj['groups']:
            item = Group()
            item.deserialize(group)
            self.groups.append(item)
        self.num_inputs = obj['num_inputs']


class Model(SerialData, Candidate):
    def __init__(self, identity_threshold: float = IDENTITY_THRESHOLD, normal_cell_n: int = NORMAL_CELL_N, cell_layers: int = CELL_LAYERS, initial_layer_dims: int = INITIAL_LAYER_DIMS, use_post_block_reduce: bool = USE_POST_BLOCK_REDUCE):
        super().__init__()
        self.blocks: List[Block] = []  # [NORMAL_CELL, REDUCTION_CELL]
        self.identity_threshold = identity_threshold
        self.normal_cell_n = normal_cell_n
        self.cell_layers = cell_layers
        self.initial_layer_dims = initial_layer_dims
        self.use_post_block_reduce = use_post_block_reduce
        self.keras_model = None
        self.model_name = 'evo_' + str(time.time())#str(time.time()).split('.')[0][4:]
        self.accuracy = 0.
        self.metrics = {
            'accuracy': 0.,
            'average_train_time': 0.,
            'average_inference_time': 0.,
            'compile_time': 0.,
            'build_time': 0.
        }

    def populate_with_NASnet_blocks(self, random_attachment: bool = False, random_op: bool = False):
        groups_in_block = 5
        ops_in_group = 2
        group_inputs = 2

        def get_block():
            block = Block(group_inputs)
            block.groups = [Group() for x in range(groups_in_block)]
            for i in range(groups_in_block):
                block.groups[i].operations = [OperationItem(i + group_inputs) for x in range(ops_in_group)]  # +2 because 2 inputs for block, range(2) because pairwise groups
                for j in range(ops_in_group):
                    block.groups[i].operations[j].actual_attachment = min(j, group_inputs - 1)

                # TODO set operations
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
            resized_input = builder.dim_change(graph_input, self.initial_layer_dims)
        previous_output = resized_input
        block_input = [resized_input, previous_output]
        for layer in range(self.cell_layers):
            for normal_cells in range(self.normal_cell_n):
                with tf.name_scope(f'normal_cell_{layer}_{normal_cells}'):
                    block_output = block_ops[0](block_input, builder)
                    block_input = [block_output, previous_output]
                    previous_output = block_output
            with tf.name_scope(f'reduction_cell_{layer}'):
                block_output = block_ops[1](block_input, builder)
            if layer != self.cell_layers - 1:
                # don't add a reduction layer at the very end of the graph before the fully connected layer
                with tf.name_scope(f'reduction_layer_{layer}'):
                    block_output = builder.downsize(block_output)
                    previous_output = builder.downsize(previous_output)
                    block_input = [block_output, previous_output]
                    previous_output = block_output

        with tf.name_scope(f'end_block'):
            output = builder.concat(block_input)
            output = builder.flatten(output)
            output = builder.dense(output, size = 64, activation=tf.nn.relu)
            output = builder.dense(output, size = 10)

        return output

    def mutate(self):
        other_mutation_threshold = ((1 - self.identity_threshold) / 2.) + self.identity_threshold

        block_index = int(np.random.random() * len(self.blocks))
        select_block = self.blocks[block_index]
        group_index = int(np.random.random() * len(select_block.groups))
        select_group = select_block.groups[group_index]
        item_index = int(np.random.random() * len(select_group.operations))
        select_item = select_group.operations[item_index]
        select_mutation = np.random.random()
        mutation_string = f'mutating block {block_index}, group {group_index}, item {item_index}: '
        if select_mutation < self.identity_threshold:
            # identity mutation
            print(mutation_string + 'identity mutation')
        elif self.identity_threshold < select_mutation < other_mutation_threshold:
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
            'metrics': {
                'accuracy': float(self.metrics['accuracy'])
            },
            'hyperparameters':
            {
                'identity_threshold': self.identity_threshold,
                'normal_cell_n': self.normal_cell_n,
                'cell_layers': self.cell_layers,
                'initial_layer_dims': self.initial_layer_dims,
                'use_post_block_reduce': self.use_post_block_reduce
            }
        }

    def deserialize(self, obj:dict) -> None:
        for block in obj['blocks']:
            item = Block()
            item.deserialize(block)
            self.blocks.append(item)
        self.metrics['accuracy'] = obj['metrics']['accuracy']
        self.identity_threshold = obj['hyperparameters']['identity_threshold']
        self.normal_cell_n = obj['hyperparameters']['normal_cell_n']
        self.cell_layers = obj['hyperparameters']['cell_layers']
        self.initial_layer_dims = obj['hyperparameters']['initial_layer_dims']
        self.use_post_block_reduce = obj['hyperparameters']['use_post_block_reduce']

    def evaluate_fitness(self, dataset: Dataset) -> None:
        self._evaluate_fitness_graph(dataset)

    def _evaluate_fitness_graph(self, dataset: Dataset) -> None:
        TRAIN_EPOCHS = 2
        TRAIN_STEPS = 1
        BATCH_SIZE = 32

        # keras_graph = tfp.keras.backend.get_session().graph
        keras_graph = tf.Graph()

        with keras_graph.as_default():
            build_time = time.time()
            print(dataset.images_shape)
            model_input = tf.keras.Input(shape=dataset.images_shape)
            model_output = self.build_graph(model_input)

            # layer = model_input
            #
            # layer = tf.keras.layers.Conv2D(32, 3, 1, 'same', activation='relu')(layer)
            # layer = tf.keras.layers.MaxPool2D((2,2))(layer)
            # layer = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(layer)
            # layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
            # layer = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(layer)
            # # layer = tf.keras.layers.BatchNormalization()(layer)
            # layer = tf.keras.layers.Flatten()(layer)
            # layer = tf.keras.layers.Dense(64, activation='relu')(layer)
            # # layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(layer)
            # layer = tf.keras.layers.Dense(10)(layer)
            #
            # model_output = layer

            self.keras_model = tf.keras.Model(inputs=model_input, outputs=model_output)


            build_time = time.time() - build_time

            tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_dir, self.model_name))

            compile_time = time.time()

            optimizer = tf.keras.optimizers.Adam(0.001)
            self.keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            compile_time = time.time() - compile_time

            train_time = time.time()
            self.keras_model.fit(dataset.train_images, dataset.train_labels, epochs=TRAIN_EPOCHS, callbacks=[tensorboard_callback])
            train_time = time.time() - train_time

            interence_time = time.time()
            evaluated_metrics = self.keras_model.evaluate(dataset.test_images, dataset.test_labels)
            inference_time = time.time() - interence_time

            self.metrics['accuracy'] = evaluated_metrics[-1]
            self.metrics['build_time'] = build_time
            self.metrics['compile_time'] = compile_time
            self.metrics['average_train_time'] = train_time / float(BATCH_SIZE * TRAIN_STEPS * TRAIN_EPOCHS)
            self.metrics['average_inference_time'] = inference_time / float(len(dataset.test_images))
            self.fitness = self.metrics['accuracy']
            # self.save()
            self.keras_model = None

    def _evaluate_fitness_eager(self, dataset: Dataset) -> None:
        # keras_graph = tfp.keras.backend.get_session().graph
        #
        # with keras_graph.as_default():
        #     model_input = tf.keras.Input(shape=dataset.images_shape)
        #     model_output = self.build_graph(model_input)
        #     self.keras_model = tf.keras.Model(inputs=model_input, outputs=model_output)
        #
        #     tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_dir, self.model_name))
        #
        #     self.keras_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        #
        #     self.keras_model.fit(dataset.train_images, dataset.train_labels, batch_size=1, epochs=1, steps_per_epoch=10, callbacks=[tensorboard_callback])
        #
        #     evaluated_metrics = self.keras_model.evaluate(dataset.test_images, dataset.test_labels)
        #     self.accuracy = evaluated_metrics[-1]
        #     self.fitness = self.accuracy
        #     # self.save()
        #     self.keras_model = None
        pass

    def save(self):
        dir_name = os.path.join(model_save_dir, self.model_name)
        os.mkdir(dir_name)
        _write_model_structure_to_json(self, dir_name, self.model_name)
        if self.keras_model is not None:
            _save_keras_model(self.keras_model, dir_name, self.model_name)

    def duplicate(self) -> Model:
        return copy.deepcopy(self)

    @staticmethod
    def load(name) -> Model:
        dir_name = os.path.join(model_save_dir, name)
        if not os.path.exists(dir_name):
            print('Model does not exist at specified location')
            return Model()

        result = _load_model_structure_from_json(dir_name, name)
        result.model_name = name

        contained_files = os.listdir(dir_name)
        contains_keras_model = False

        for fl in contained_files:
            if len(fl) > 3 and fl[-3:] == '.h5':
                contains_keras_model = True

        if contains_keras_model:
            result.keras_model = _load_keras_model(dir_name, name)

        return result

    @staticmethod
    def load_most_recent_model() -> Model:
        models = os.listdir(model_save_dir)
        if len(models) == 0:
            print('No saved models exist to load')
            return Model()

        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # sort models by a number attached to the end of their name
        name = models[-1]
        return Model.load(name)


def _write_model_structure_to_json(model: Model, dir: str, name: str) -> None:
    serialized = model.serialize()
    with open(os.path.join(dir, name + '.json'), 'w') as fl:
        json.dump(serialized, fl, indent=4)


def _load_model_structure_from_json(dir: str, name: str) -> Model:
    with open(os.path.join(dir, name + '.json'), 'r') as fl:
        serialized = json.load(fl)
    model = Model()
    model.deserialize(serialized)
    return model


def _load_keras_model(dir: str, name: str):
    model = load_model(os.path.join(dir, name + '.h5'))
    return model


def _save_keras_model(model, dir: str, model_name: str):
    model.save(os.path.join(dir, model_name + '.h5'))


def do_test():
    model = Model(IDENTITY_THRESHOLD, NORMAL_CELL_N, CELL_LAYERS, INITIAL_LAYER_DIMS, USE_POST_BLOCK_REDUCE)
    model.populate_with_NASnet_blocks()

    for x in range(100):
        model.mutate()

    dataset = Dataset.get_build_set()

    model.evaluate(dataset)

    # writer = tf.summary.create_file_writer('../res/')
    #
    # keras_graph = tfp.keras.backend.get_session().graph
    #
    # with keras_graph.as_default():
    #
    #     model_input = tf.keras.Input(shape=[16, 16, 3])
    #     model_output = model_obj.build_graph(model_input)
    #     model = tf.keras.Model(inputs=model_input, outputs=model_output)
    #
    #
    #
    #     model_name = 'evo_' + str(time.time())[4:-4]
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_dir, model_name))
    #
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     print(f'----{model_name}')
    #
    #     model.fit(train_images, train_labels, batch_size=1, epochs=1, callbacks=[tensorboard_callback])
    #
    #     evaluated_metrics = model.evaluate(test_images, test_labels)
    #     model_obj.accuracy = evaluated_metrics[-1]
    #     print(evaluated_metrics)
    #
    #     _save_model(model_obj, model)
    #
    # loaded_model_structure, loaded_keras_model = _load_most_recent_model()


if __name__ == '__main__':
    do_test()
