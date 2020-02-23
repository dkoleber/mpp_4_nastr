from __future__ import annotations
import numpy as np
import tensorflow.python as tfp
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Tuple
from enum import Enum, IntEnum
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))

res_dir = os.path.join(HERE,'..\\res\\')
tensorboard_dir = os.path.join(HERE,'..\\tensorboard\\')

tf.compat.v1.disable_eager_execution()
'''
TODO:
- random selection of re-attachments with exclusions for existing attachments
- initial layer resizing
- activations
- no final reduction layer at end


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
NORMAL_CELL_N = 3
CELL_LAYERS = 3

OP_DIMS = 8

#
# class Builder:
#     @staticmethod
#     def get_op(op_type: OperationType, op_input):
#         if op_type == OperationType.IDENTITY:
#             return tf.identity(op_input)
#         elif op_type == OperationType.CONV3x3:
#             return tf.compat.v1.layers.conv2d(op_input, 8, 1, 1, 'same')
#
#     @staticmethod
#     def add(values):
#         return tf.add_n(values)
#
#     @staticmethod
#     def concat(values):
#         return tf.concat(values, axis=3)
#
#     @staticmethod
#     def dense(layer_input):
#         return tf.compat.v1.layers.dense(layer_input, 10)
#
#     @staticmethod
#     def flatten(layer_input):
#         return tf.compat.v1.layers.flatten(layer_input)
#
#     @staticmethod
#     def softmax(layer_input):
#         return tf.nn.softmax(layer_input)


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
        if op_type == OperationType.SEP_3X3:
            return tf.keras.layers.SeparableConv2D(input_shape, 3, 1, 'same', name=self.get_name('SEP_3X3'))(op_input)
        elif op_type == OperationType.SEP_5X5:
            return tf.keras.layers.SeparableConv2D(input_shape, 5, 1, 'same', name=self.get_name('SEP_5X5'))(op_input)
        elif op_type == OperationType.SEP_7X7:
            return tf.keras.layers.SeparableConv2D(input_shape, 7, 1, 'same', name=self.get_name('SEP_7X7'))(op_input)
        elif op_type == OperationType.AVG_3X3:
            return tf.keras.layers.AveragePooling2D(3, 1, 'same', name=self.get_name('AVG_3X3'))(op_input)
        elif op_type == OperationType.MAX_3X3:
            return tf.keras.layers.MaxPool2D(3, 1, 'same', name=self.get_name('MAX_3X3'))(op_input)
        elif op_type == OperationType.DIL_3X3:
            return tf.keras.layers.Conv2D(input_shape, 3, 1, 'same', name=self.get_name('DIL_3X3'), dilation_rate=2)(op_input)
        elif op_type == OperationType.SEP_1X7_7X1:
            layer = tf.keras.layers.Conv2D(input_shape, (1, 7), 1, 'same', name=self.get_name('SEP_1X7'))(op_input)
            return tf.keras.layers.Conv2D(input_shape, (7, 1), 1, 'same', name=self.get_name('SEP_7X1'))(layer)
        else: # OperationType.IDENTITY and everything else
            return self.identity(op_input, input_shape)

    def add(self, values):
        return tf.keras.layers.Add(name=self.get_name('add'))(values)

    def concat(self, values, scope=''):
        if len(values) == 1:
            return self.identity(values[0])
        else:
            return tf.keras.layers.Concatenate(axis=3, name=self.get_name('concat'))(values)
        # return tf.keras.layers.Add(name=self.get_name('fakeconcat'))(values) #TODO: change to concat?

    def dense(self, layer_input, scope=''):
        return tf.keras.layers.Dense(10, name=self.get_name('dense'))(layer_input)

    def flatten(self, layer_input, scope=''):
        return tf.keras.layers.Flatten(name=self.get_name('flatten'))(layer_input)

    def softmax(self, layer_input):
        return tf.keras.layers.Softmax(name=self.get_name('softmax'))(layer_input)

    def identity(self, layer_input, output_size=1):
        # return tf.identity(layer_input, name=self.get_name('identity'))  # don't use- messes with scoping
        # input_size = layer_input.shape[3]
        # if output_size != 1 and input_size == 1:
        #     return tf.keras.layers.Concatenate(axis=3, name=self.get_name('identity'))([layer_input for x in range(output_size)])
        # elif input_size != 1 and input_size != output_size:
        #     return self.dim_redux_conv(layer_input, output_size)
        # else:
        return tf.keras.layers.Lambda(lambda x: x, name=self.get_name('identity'))(layer_input)

    def dim_change(self, layer_input, output_size):
        return tf.keras.layers.Conv2D(output_size, 1, 1, 'same', name=self.get_name('dim_redux_1x1'))(layer_input)

    def downsize(self, layer_input):
        return tf.keras.layers.Conv2D(1, 3, 2, 'same', name=self.get_name('downsize'))(layer_input)




class SerialData(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def deserialize(self, obj:dict) -> None:
        pass






class OperationItem:
    def __init__(self, attachment_index: int):
        self.operation_type: OperationType = OperationType.IDENTITY
        self.attachment_index: int = attachment_index
        self.actual_attachment: int = 0

    def build_operation(self, operation_input, builder: Builder):
        return builder.get_op(self.operation_type, operation_input)


class Group:
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


class Block:
    def __init__(self, num_inputs: int):
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
        return builder.concat(unattached)


class Model:
    def __init__(self, ):
        self.blocks: List[Block] = []  # [NORMAL_CELL, REDUCTION_CELL]

    def populate_with_NASnet_blocks(self):
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
        self.blocks.append(get_block())  # normal block
        self.blocks.append(get_block())  # reduction block

    def build_graph(self, graph_input):
        builder = Builder()

        block_ops = []
        for block in self.blocks:
            block_ops.append(block.build_block)

        previous_output = graph_input
        block_input = [graph_input, previous_output]
        for layer in range(CELL_LAYERS):
            for normal_cells in range(NORMAL_CELL_N):
                with tf.name_scope(f'normal_cell_{layer}_{normal_cells}'):
                    block_output = block_ops[0](block_input, builder)
                    block_input = [block_output, previous_output]
                    previous_output = block_output
            with tf.name_scope(f'reduction_cell_{layer}'):
                block_output = block_ops[1](block_input, builder)
            with tf.name_scope(f'reduction_layer_{layer}'):
                block_output = builder.downsize(block_output)
                previous_output = builder.downsize(previous_output)
                block_input = [block_output, previous_output]
                previous_output = block_output


        with tf.name_scope(f'end_block'):
            output = builder.concat(block_input)
            output = builder.flatten(output)
            output = builder.dense(output)
            output = builder.softmax(output)


        return output

    def mutate(self):

        other_mutation_threshold = ((1 - IDENTITY_THRESHOLD) / 2.) + IDENTITY_THRESHOLD

        block_index = int(np.random.random() * len(self.blocks))
        select_block = self.blocks[block_index]
        group_index = int(np.random.random() * len(select_block.groups))
        select_group = select_block.groups[group_index]
        item_index = int(np.random.random() * len(select_group.operations))
        select_item = select_group.operations[item_index]
        select_mutation = np.random.random()
        print(f'--mutating block {block_index}, group {group_index}, item {item_index}')
        if select_mutation < IDENTITY_THRESHOLD:
            print('identity mutation')
            return
        elif IDENTITY_THRESHOLD < select_mutation < other_mutation_threshold:
            # hidden state mutation = change inputs
            if group_index != 0:
                previous_attachment = select_item.actual_attachment
                select_item.actual_attachment = int(np.random.random() * select_item.attachment_index)
                print(f'mutating hidden state from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(f'skipping state mutation for group 0')

        else:
            previous_op = select_item.operation_type
            select_item.operation_type = int(np.random.random() * (OperationType.SEP_1X7_7X1 + 1))
            print(f'mutating operation type from {previous_op} to {select_item.operation_type}')



def do_test():
    model_obj = Model()
    model_obj.populate_with_NASnet_blocks()

    for x in range(100):
        model_obj.mutate()

    # writer = tf.summary.create_file_writer('../res/')
    keras_graph = tfp.keras.backend.get_session().graph

    with keras_graph.as_default():

        model_input = tf.keras.Input(shape=[16, 16, 3])
        model_output = model_obj.build_graph(model_input)
        model = tf.keras.Model(inputs=model_input, outputs=model_output)

        train_images = np.zeros([4, 16, 16, 3])
        train_labels = np.zeros([4, 10])

        model_name = 'evo_' + str(time.time())[4:-4]
        tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_dir, model_name))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print(f'----{model_name}')

        model.fit(train_images, train_labels, batch_size=1, epochs=1, callbacks=[tensorboard_callback])


if __name__ == '__main__':
    do_test()
