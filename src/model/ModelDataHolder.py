from __future__ import annotations
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from model.MetaModel import *
from model.DropPath import *


class ModelParsingHelper:
    def __init__(self):
        self.counts = {
            'identity_operation':0,
            'identity_reduction_operation': 0,
            'seperable_convolution_operation': 0,
            'average_pooling_operation': 0,
            'max_pooling_operation': 0,
            'doubly_seperable_convolution_operation': 0,
            'batch_normalization': 0,
            'add': 0,
            'conv2d': 0,
            'concatenate': 0,
            'dimensionality_change_operation': 0,
            'relu6_layer': 0,
            'lambda': 0,
            'dense_operation': 0
        }

    @staticmethod
    def get_op_name(op:OperationType, requires_reduction: bool = False):
        if op == OperationType.IDENTITY:
            if requires_reduction:
                return 'identity_reduction_operation'
            else:
                return 'identity_operation'

        vals = {
            OperationType.SEP_3X3: 'seperable_convolution_operation',
            OperationType.SEP_5X5: 'seperable_convolution_operation',
            OperationType.SEP_7X7: 'seperable_convolution_operation',
            OperationType.DIL_3X3: 'seperable_convolution_operation',
            OperationType.AVG_3X3: 'average_pooling_operation',
            OperationType.MAX_3X3: 'max_pooling_operation',
            OperationType.SEP_1X7_7X1: 'doubly_seperable_convolution_operation'
        }
        return vals[op]

    @staticmethod
    def _get_layer_name(name, number):
        if number == 0:
            return name
        else:
            return f'{name}_{number}'

    def get_next_name(self, name):
        if name not in self.counts:
            return name
        else:
            result = ModelParsingHelper._get_layer_name(name, self.counts[name])
            self.counts[name] += 1
            return result

    def get_next_name_for_op(self, op, input_dim: int = 1, output_dim: int = 1):
        layer = KerasOperationFactory.get_operation(op, input_dim, output_dim)
        layer.add_self_to_parser_counts(self)
        return self.get_next_name(ModelParsingHelper.get_op_name(op, input_dim != output_dim))


class GroupDataHolder:
    def __init__(self, input_dim: int, target_dim: int, meta_group: MetaGroup, drop_path_tracker: DropPathTracker, cell_position_as_ratio: float, parser: ModelParsingHelper = None, keras_model:tf.keras.models.Model = None):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.drop_path_tracker = drop_path_tracker

        self.ops = []
        self.attachments: List[int] = []

        self.op_types = [x.operation_type for x in meta_group.operations]

        if keras_model is None:

            for op in meta_group.operations:
                self.ops.append(KerasOperationFactory.get_operation(op.operation_type, self.input_dim, self.target_dim))
                self.attachments.append(op.actual_attachment)

            self.addition_layer = tf.keras.layers.Add()
            self.drop_path_ops = [
                DropPathOperation(cell_position_as_ratio, drop_path_tracker),
                DropPathOperation(cell_position_as_ratio, drop_path_tracker)
            ]


        else:

            for op in meta_group.operations:
                if op.operation_type == OperationType.IDENTITY and input_dim == target_dim:
                    self.ops.append(KerasOperationFactory.get_operation(OperationType.IDENTITY, self.input_dim, self.target_dim))
                else:
                    self.ops.append(keras_model.get_layer(parser.get_next_name_for_op(op.operation_type, self.input_dim, self.target_dim)))
                self.attachments.append(op.actual_attachment)

            self.addition_layer = keras_model.get_layer(parser.get_next_name('add'))
            self.drop_path_ops = [
                DropPathOperation(cell_position_as_ratio, drop_path_tracker),
                DropPathOperation(cell_position_as_ratio, drop_path_tracker)
            ] #TODO

    def build(self, inputs):
        results = []
        for index, obj in enumerate(self.ops):
            op = obj(inputs[self.attachments[index]])
            if self.op_types[index] != OperationType.IDENTITY:
                op = self.drop_path_ops[index](op)
            results.append(op)

        return self.addition_layer(results)


class CellDataHolder:
    def __init__(self, input_dim: int, target_dim: int, meta_cell: MetaCell, reduce_current: bool, drop_path_tracker: DropPathTracker, cell_position_as_ratio, parser:ModelParsingHelper = None, keras_model:tf.keras.models.Model = None):
        self.target_dim = target_dim
        self.input_dim = input_dim
        self.reduce_current = reduce_current

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
                self.groups.append(GroupDataHolder(self.input_dim, self.target_dim, group, drop_path_tracker, cell_position_as_ratio))
                for op in group.operations:
                    used_group_indexes.append(op.actual_attachment)

        else:
            for group in meta_cell.groups:
                self.groups.append(GroupDataHolder(self.input_dim, self.target_dim, group, drop_path_tracker, cell_position_as_ratio, parser, keras_model))
                for op in group.operations:
                    used_group_indexes.append(op.actual_attachment)

        self.unused_group_indexes: List[int] = [x for x in range(len(meta_cell.groups) + 2) if x not in used_group_indexes]
        self.output_dim = target_dim * len(self.unused_group_indexes)

        self.post_dim_change = None
        self.post_concat = None

        if keras_model is None:
            if len(self.unused_group_indexes) > 1:
                self.post_concat = tf.keras.layers.Concatenate(axis=3)

            if not self.reduce_current and self.output_dim != self.input_dim: #reduce residual to current output
                self.post_dim_change = DimensionalityChangeOperation(self.input_dim, self.output_dim)

            if self.reduce_current and self.output_dim != self.target_dim:
                self.post_dim_change = DimensionalityChangeOperation(self.output_dim, self.target_dim)
                self.output_dim = self.target_dim

        else:
            if len(self.unused_group_indexes) > 1:
                self.post_concat = keras_model.get_layer(parser.get_next_name('concatenate'))

            if (not self.reduce_current and self.output_dim != self.input_dim) or (self.reduce_current and self.output_dim != self.target_dim):
                try:
                    self.post_dim_change = keras_model.get_layer(parser.get_next_name('dimensionality_change_operation'))
                    self.post_dim_change.add_self_to_parser_counts(parser)

                except:
                    # this can occur if this is the last cell in the network.
                    # Since the reduced residual isn't actually used in the output, then it's not serialized with the model
                    # and therefore can't be loaded. However, we still need to account for its name,
                    # since it still existed (and impacted subsequent op names) prior to the model being serialized.
                    temp_dim = DimensionalityChangeOperation(1, 1)
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


class ReductionCellDataHolder(CellDataHolder):
    def __init__(self, input_dim: int, target_dim: int, meta_cell: MetaCell, reduce_current: bool, drop_path_tracker: DropPathTracker, cell_position_as_ratio, parser:ModelParsingHelper = None, keras_model:tf.keras.models.Model = None, expansion_factor: int = 1, reduce_before: bool = False):
        self.expansion_factor = expansion_factor
        self.reduce_before = reduce_before

        target_dim_to_pass = target_dim
        if reduce_before:
            target_dim_to_pass *= self.expansion_factor

        super().__init__(input_dim, target_dim_to_pass, meta_cell, reduce_current, drop_path_tracker, cell_position_as_ratio, parser, keras_model)

        if keras_model is None: #TODO: factorized reduction
            if self.expansion_factor != 1:
                if self.reduce_before:
                    self.dim_change_current = DimensionalityChangeOperation(self.input_dim, self.target_dim)
                    self.dim_change_previous = DimensionalityChangeOperation(self.input_dim, self.target_dim)
                else:
                    expanded_dim = self.output_dim * self.expansion_factor
                    self.dim_change_current = DimensionalityChangeOperation(self.output_dim, expanded_dim)
                    self.dim_change_previous = DimensionalityChangeOperation(self.output_dim, expanded_dim)
                    self.output_dim = expanded_dim

            self.reduce_current = tf.keras.layers.Conv2D(self.output_dim, 3, 2, 'same')
            self.reduce_previous = tf.keras.layers.Conv2D(self.output_dim, 3, 2, 'same')

        else:
            if self.expansion_factor != 1:
                self.dim_change_current = keras_model.get_layer(parser.get_next_name('dimensionality_change_operation'))
                self.dim_change_previous = keras_model.get_layer(parser.get_next_name('dimensionality_change_operation'))
                self.dim_change_current.add_self_to_parser_counts(parser)
                self.dim_change_previous.add_self_to_parser_counts(parser)

            if not self.reduce_before:
                expanded_dim = self.output_dim * self.expansion_factor
                self.output_dim = expanded_dim

            self.reduce_current = keras_model.get_layer(parser.get_next_name('conv2d'))
            self.reduce_previous = keras_model.get_layer(parser.get_next_name('conv2d'))


    def build(self, inputs):
        # inputs are [current values, residual/previous values]
        values = inputs

        if self.reduce_before and self.expansion_factor != 1:
            values = [self.dim_change_current(values[0]), self.dim_change_previous(values[1])]

        values = super().build(values)

        if not self.reduce_before and self.expansion_factor != 1:
            values = [self.dim_change_current(values[0]), self.dim_change_previous(values[1])]

        values = [self.reduce_current(values[0]), self.reduce_previous(values[1])]
        return values


class ModelDataHolder:
    def __init__(self, meta_model:MetaModel, keras_model:tf.keras.models.Model = None):
        if len(meta_model.cells) == 0:
            print('Error: no cells in meta model. Did you forget to populate it with cells?')
            return

        self.expansion_factor = meta_model.hyperparameters.parameters['REDUCTION_EXPANSION_FACTOR']
        self.reduce_current = meta_model.hyperparameters.parameters['REDUCE_CURRENT']
        self.reduction_expansion_before = meta_model.hyperparameters.parameters['REDUCTION_EXPANSION_BEFORE']
        target_dim = meta_model.hyperparameters.parameters['TARGET_FILTER_DIMS']
        previous_dim = target_dim

        num_cell_layers = meta_model.hyperparameters.parameters['CELL_LAYERS']
        num_normal_cells_per_layer = meta_model.hyperparameters.parameters['NORMAL_CELL_N']


        steps_per_epoch = math.ceil(50000 / meta_model.hyperparameters.parameters['BATCH_SIZE']) #TODO: MAGIC NUMBER
        steps_so_far = (len(meta_model.metrics.metrics['accuracy']) * meta_model.hyperparameters.parameters['TRAIN_EPOCHS']) * steps_per_epoch
        total_steps = meta_model.hyperparameters.parameters['TRAIN_ITERATIONS'] * meta_model.hyperparameters.parameters['TRAIN_EPOCHS'] * steps_per_epoch
        self.drop_path_tracker = DropPathTracker(meta_model.hyperparameters.parameters['DROP_PATH_CHANCE'], steps_so_far, total_steps)

        self.cells: List[CellDataHolder] = []

        current_cell_count = 0
        total_cells_count = (num_cell_layers * num_normal_cells_per_layer) + (num_cell_layers - 1)

        def get_cell_position_as_ratio():
            nonlocal current_cell_count
            nonlocal total_cells_count
            result = (current_cell_count + 1) / total_cells_count
            current_cell_count += 1
            return result

        if keras_model is None:
            self.initial_resize = tf.keras.layers.Conv2D(target_dim, 3, 1, 'valid')
            self.initial_norm = tf.keras.layers.BatchNormalization()

            for layer in range(num_cell_layers):
                for normal_cells in range(num_normal_cells_per_layer):
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[0], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio())
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                if layer != num_cell_layers - 1:
                    cell = ReductionCellDataHolder(previous_dim, target_dim, meta_model.cells[1], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(),
                                                   expansion_factor=self.expansion_factor,
                                                   reduce_before=self.reduction_expansion_before)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                    target_dim *= self.expansion_factor

            self.final_flatten = tf.keras.layers.Flatten()
            self.final_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))
            self.final_activation = Relu6Layer()
            # self.final_dropout = tf.keras.layers.Dropout(meta_model.hyperparameters.parameters['DROPOUT_RATE'])
            self.final_dense = DenseOperation(target_dim, 10, .5) #TODO: remove dropout chance?
        else:
            parser = ModelParsingHelper()

            self.initial_resize = keras_model.get_layer(parser.get_next_name('conv2d'))
            self.initial_norm = keras_model.get_layer(parser.get_next_name('batch_normalization'))

            for layer in range(num_cell_layers):
                for normal_cells in range(num_normal_cells_per_layer):
                    cell = CellDataHolder(previous_dim, target_dim, meta_model.cells[0], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(),
                                          parser,
                                          keras_model)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                if layer != num_cell_layers - 1:
                    cell = ReductionCellDataHolder(previous_dim, target_dim, meta_model.cells[1], self.reduce_current, self.drop_path_tracker, get_cell_position_as_ratio(),
                                                   parser,
                                                   keras_model,
                                                   expansion_factor=self.expansion_factor,
                                                   reduce_before=self.reduction_expansion_before)
                    self.cells.append(cell)
                    previous_dim = cell.output_dim
                    target_dim *= self.expansion_factor


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