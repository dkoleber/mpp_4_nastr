from __future__ import annotations

import math
import os
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from model.MetaModel import *
from model.DropPath import *
from model.CustomLayers import *

class ModelParsingHelper:
    def __init__(self):
        self.counts = {}

    @staticmethod
    def get_op_name(op:OperationType, requires_reduction: bool = False):
        return OperationType.lookup_opname(op, requires_reduction)

    @staticmethod
    def _get_layer_name(name, number):
        if number == 0:
            return name
        else:
            return f'{name}_{number}'

    def get_next_name(self, name):
        if name not in self.counts:
            self.counts[name] = 1
            return name
        else:
            result = ModelParsingHelper._get_layer_name(name, self.counts[name])
            self.counts[name] += 1
            return result

    def get_next_name_for_op(self, op, input_dim: int, output_dim: int, stride: int):
        layer = KerasOperationFactory.get_operation(op, input_dim, output_dim, stride)
        layer.add_self_to_parser_counts(self)
        return self.get_next_name(ModelParsingHelper.get_op_name(op, (input_dim != output_dim or stride != 1)))


class GroupDataHolder:
    def __init__(self,
                 input_dim: int,
                 target_dim: int,
                 meta_group: MetaGroup,
                 drop_path_tracker: DropPathTracker,
                 cell_position_as_ratio: float,
                 stride: int = 1,
                 parser: ModelParsingHelper = None,
                 keras_model:tf.keras.models.Model = None):
        self.cell_input_dim = input_dim
        self.target_dim = target_dim
        self.drop_path_tracker = drop_path_tracker
        self.cell_stride = stride
        self.num_ops = len(meta_group.operations)

        self.ops = []
        self.attachments = [x.actual_attachment for x in meta_group.operations]

        if keras_model is None:
            for index, op in enumerate(meta_group.operations):
                self.ops.append(KerasOperationFactory.get_cell_operation(op.operation_type, self.cell_input_dim, self.target_dim, self.cell_stride, op.actual_attachment))

            self.addition_layer = tf.keras.layers.Add()
            self.drop_path_ops = [DropPathOperation(cell_position_as_ratio, drop_path_tracker) for _ in range(self.num_ops)]

        else:

            for index, op in enumerate(meta_group.operations):
                actual_input_dim = self.cell_input_dim if op.actual_attachment < 2 else self.target_dim
                actual_stride = self.cell_stride if op.actual_attachment < 2 else 1
                if op.operation_type == OperationType.IDENTITY and actual_input_dim == target_dim and actual_stride == 1:
                    self.ops.append(KerasOperationFactory.get_operation(OperationType.IDENTITY, actual_input_dim, self.target_dim, self.cell_stride))
                    parser.get_next_name('identity_operation')
                else:
                    self.ops.append(keras_model.get_layer(parser.get_next_name_for_op(op.operation_type, actual_input_dim, self.target_dim, self.cell_stride)))

            self.addition_layer = keras_model.get_layer(parser.get_next_name('add'))
            self.drop_path_ops = [DropPathOperation(cell_position_as_ratio, drop_path_tracker) for _ in range(self.num_ops)]

    def build(self, inputs):
        results = []
        for index, obj in enumerate(self.ops):
            op = obj(inputs[self.attachments[index]])
            if type(obj) is not IdentityOperation:
                op = self.drop_path_ops[index](op)
            results.append(op)

        # shapes = [x.shape.as_list()[1:] for x in inputs]
        # names = [x.__repr__() for x in self.ops]
        # print(f'attachments: {self.attachments}, ops: {names}, target: {self.target_dim}, {shapes}')

        return self.addition_layer(results)


class CellDataHolder:
    def __init__(self,
                 input_dim: int,
                 target_dim: int,
                 meta_cell: MetaCell,
                 reduce_current: bool,
                 drop_path_tracker: DropPathTracker,
                 cell_position_as_ratio,
                 stride: int = 1,
                 parser: ModelParsingHelper = None,
                 keras_model: tf.keras.models.Model = None):
        self.target_dim = target_dim
        self.input_dim = input_dim
        self.reduce_current = reduce_current
        self.stride = stride

        # target dim is the dimension that we want the groups to be outputting
        # input dim is the dimension that the previous cell produced
        # output dim is the dimension that this cell produces
        #
        # the target dim and output dim are the same if there's only one unused cell
        # the output dim and input dim are the same if the previous cell is the same type of cell
        # all three are the same if the previous two statements are true

        self.groups: List[GroupDataHolder] = []

        used_group_indexes: List[int] = [0, 1]  # last cell, cell before that

        if keras_model is None:
            for group in meta_cell.groups:
                self.groups.append(GroupDataHolder(self.input_dim, self.target_dim, group, drop_path_tracker, cell_position_as_ratio, self.stride))
                for op in group.operations:
                    used_group_indexes.append(op.actual_attachment)

        else:
            for group in meta_cell.groups:
                self.groups.append(GroupDataHolder(self.input_dim, self.target_dim, group, drop_path_tracker, cell_position_as_ratio, self.stride, parser, keras_model))
                for op in group.operations:
                    used_group_indexes.append(op.actual_attachment)

        self.unused_group_indexes: List[int] = [x for x in range(len(meta_cell.groups) + 2) if x not in used_group_indexes]
        self.output_dim = target_dim * len(self.unused_group_indexes)

        self.post_dim_change = None
        self.post_concat = None

        if keras_model is None:
            if len(self.unused_group_indexes) > 1:
                self.post_concat = tf.keras.layers.Concatenate(axis=3)

            if not self.reduce_current and (self.stride != 1 or self.output_dim != self.input_dim): #reduce residual to current output

                self.post_dim_change = ConvolutionalOperation(self.input_dim, self.output_dim, self.stride)

            if self.reduce_current and (self.stride != 1 or self.output_dim != self.target_dim):
                self.post_dim_change = ConvolutionalOperation(self.output_dim, self.target_dim, self.stride)
                self.output_dim = self.target_dim

        else:
            if len(self.unused_group_indexes) > 1:
                self.post_concat = keras_model.get_layer(parser.get_next_name('concatenate'))

            if self.stride != 1 or (not self.reduce_current and self.output_dim != self.input_dim) or (self.reduce_current and self.output_dim != self.target_dim):
                try:
                    self.post_dim_change = keras_model.get_layer(OperationType.lookup_opname(OperationType.CONV_3X3))
                    self.post_dim_change.add_self_to_parser_counts(parser)

                except:
                    # this can occur if this is the last cell in the network.
                    # Since the reduced residual isn't actually used in the output, then it's not serialized with the model
                    # and therefore can't be loaded. However, we still need to account for its name,
                    # since it still existed (and impacted subsequent op names) prior to the model being serialized.
                    temp_dim = ConvolutionalOperation(1, 1)
                    temp_dim.add_self_to_parser_counts(parser)
                    pass

                if self.reduce_current:
                    self.output_dim = self.target_dim



    def build(self, inputs):
        available_inputs = [x for x in inputs]
        for group in self.groups:
            group_output = group.build(available_inputs)
            available_inputs.append(group_output)

        result = None

        if len(self.unused_group_indexes) > 1:
            concat_groups = [available_inputs[x] for x in self.unused_group_indexes]
            result = self.post_concat(concat_groups)
        else:
            result = available_inputs[-1]

        previous_result = None

        if not self.reduce_current and self.post_dim_change is not None:
            previous_result = self.post_dim_change(inputs[0])
        else:
            previous_result = inputs[0]

        if self.reduce_current and self.post_dim_change is not None:
            result = self.post_dim_change(result)

        results = [result, previous_result]
        return results


class ModelDataHolder:
    def __init__(self, meta_model:MetaModel, keras_model:tf.keras.models.Model = None):
        if len(meta_model.cells) == 0:
            print('Error: no cells in meta model. Did you forget to populate it with cells?')
            return

        self.expansion_factor = meta_model.hyperparameters.parameters['REDUCTION_EXPANSION_FACTOR']
        self.reduce_current = meta_model.hyperparameters.parameters['REDUCE_CURRENT']
        target_dim = meta_model.hyperparameters.parameters['TARGET_FILTER_DIMS']
        previous_dim = target_dim

        self.num_cell_layers = meta_model.hyperparameters.parameters['CELL_LAYERS']
        self.num_normal_cells_per_layer = meta_model.hyperparameters.parameters['NORMAL_CELL_N']

        epochs_so_far = len(meta_model.metrics.metrics['accuracy']) * meta_model.hyperparameters.parameters['TRAIN_EPOCHS']
        total_epochs = meta_model.hyperparameters.parameters['TRAIN_EPOCHS'] * meta_model.hyperparameters.parameters['TRAIN_ITERATIONS']

        self.drop_path_tracker = DropPathTracker(meta_model.hyperparameters.parameters['DROP_PATH_CHANCE'],
                                                 epochs_so_far,
                                                 total_epochs,
                                                 meta_model.hyperparameters.parameters['DROP_PATH_TOTAL_STEPS_MULTI'])

        self.cells: List[CellDataHolder] = []

        current_cell_count = 0
        total_cells_count = (self.num_cell_layers * self.num_normal_cells_per_layer) + (self.num_cell_layers - 1)

        def get_cell_position_as_ratio():
            nonlocal current_cell_count
            nonlocal total_cells_count
            result = (current_cell_count + 1) / total_cells_count
            current_cell_count += 1
            return result

        if keras_model is None:
            self.initial_resize = tf.keras.layers.Conv2D(target_dim, 3, 1, 'same')
            self.initial_norm = tf.keras.layers.BatchNormalization()

            for layer in range(self.num_cell_layers):
                for normal_cells in range(self.num_normal_cells_per_layer):
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[0], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio())
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                if layer != self.num_cell_layers - 1:
                    target_dim *= self.expansion_factor
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[1], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(), 2)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim

            self.final_flatten = tf.keras.layers.Flatten()
            self.final_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))
            self.final_activation = Relu6Layer()
            # self.final_dropout = tf.keras.layers.Dropout(meta_model.hyperparameters.parameters['DROPOUT_RATE'])
            self.final_dense = DenseOperation(target_dim, 10, .5) #TODO: remove dropout chance?
        else:
            parser = ModelParsingHelper()

            self.initial_resize = keras_model.get_layer(parser.get_next_name('conv2d'))
            self.initial_norm = keras_model.get_layer(parser.get_next_name('batch_normalization'))

            for layer in range(self.num_cell_layers):
                for normal_cells in range(self.num_normal_cells_per_layer):
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[0], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(),
                                          parser=parser,
                                          keras_model=keras_model)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                if layer != self.num_cell_layers - 1:
                    target_dim *= self.expansion_factor
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[1], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(), 2,
                                          parser=parser,
                                          keras_model=keras_model)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim

            self.final_flatten = tf.keras.layers.Flatten()
            self.final_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))
            self.final_activation = keras_model.get_layer(parser.get_next_name('relu6_layer'))
            self.final_dense = keras_model.get_layer(parser.get_next_name('dense_operation'))

    def get_hashes(self):
        hash_list = []
        for index, cell in enumerate(self.cells):
            combined = ''
            for group in cell.groups:
                for operation in group.ops:
                    # print(operation.get_weights())
                    combined += str(hash(str(operation.get_weights())))
                for attachment in group.ops:
                    combined += str(hash(attachment))
            hash_list.append(f'{index}: {hash(combined)}')
        return hash_list

    def operation_mutation(self, cell_index: int, group_index: int, operation_index: int, new_operation: int):
        actual_cell_index = 0

        def mutate_layer(index):

            previous_input_shape = self.cells[index].groups[group_index].ops[operation_index].get_input_shape_at(0)

            cell_input_dim = self.cells[index].groups[group_index].cell_input_dim
            cell_stride = self.cells[index].groups[group_index].cell_stride
            target_dim = self.cells[index].groups[group_index].target_dim

            actual_attachment = self.cells[index].groups[group_index].attachments[operation_index]
            self.cells[index].groups[group_index].ops[operation_index] = KerasOperationFactory.get_cell_operation(new_operation, cell_input_dim, target_dim, cell_stride, actual_attachment)
            self.cells[index].groups[group_index].ops[operation_index].build(previous_input_shape)
            print(f'--finished building mutated layer (operation) stride {cell_stride} shape {previous_input_shape}')

        for layer in range(self.num_cell_layers):
            for normal_cells in range(self.num_normal_cells_per_layer):
                if cell_index == 0:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1
            if layer != self.num_cell_layers - 1:
                if cell_index == 1:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1

    def hidden_state_mutation(self, cell_index: int, group_index: int, operation_index: int, new_hidden_state: int, operation_type: int):
        actual_cell_index = 0

        def mutate_layer(index):
            cell_stride = self.cells[index].groups[group_index].cell_stride
            selected_stride = cell_stride if new_hidden_state < 2 else 1

            cell_input_dim = self.cells[index].groups[group_index].cell_input_dim
            target_dim = self.cells[index].groups[group_index].target_dim
            selected_input_dim = cell_input_dim if new_hidden_state < 2 else target_dim

            cell_input_shape = list(self.cells[index].groups[0].ops[0].get_input_shape_at(0))[:-1]
            cell_output_shape = list(self.cells[index].groups[0].ops[0].get_output_shape_at(0))[:-1]
            selected_shape = cell_input_shape if new_hidden_state < 2 else cell_output_shape

            full_shape = selected_shape.copy()
            full_shape.append(selected_input_dim)

            self.cells[index].groups[group_index].ops[operation_index] = KerasOperationFactory.get_operation(operation_type, selected_input_dim, target_dim, selected_stride)
            self.cells[index].groups[group_index].ops[operation_index].build(full_shape)
            self.cells[index].groups[group_index].attachments[operation_index] = new_hidden_state
            print(f'--finished building mutated layer (state) target: {target_dim}, previous input shape: {full_shape}')

        for layer in range(self.num_cell_layers):
            for normal_cells in range(self.num_normal_cells_per_layer):
                if cell_index == 0:
                    mutate_layer(actual_cell_index)
                actual_cell_index += 1
            if layer != self.num_cell_layers - 1:
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