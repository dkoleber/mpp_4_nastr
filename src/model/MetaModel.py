from __future__ import annotations

from typing import List
import time

import copy

from graphviz import Digraph
import os

HERE = os.path.dirname(os.path.abspath(__file__))

from FileManagement import *
from Metrics import Metrics
from OperationType import OperationType
from model.SGDR import SGDR
from SerialData import SerialData
from Hyperparameters import Hyperparameters
from Dataset import ImageDataset
from model.ModelUtilities import *
from model.ModelDataHolder import *

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
        self.static_data = {}

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

    def get_unused_group_indexes(self):
        used = [x + 2 for x in range(len(self.groups))]
        for group in self.groups:
            for op in group.operations:
                if op.actual_attachment in used:
                    used.remove(op.actual_attachment)

        return used

    def process_stuff(self):



        # get residual ratios
        self.groups.sort(key=lambda x: x.operations[0].attachment_index)
        for g in self.groups:
            g.static_data['residual_ratio'] = 0
        for g in self.groups:
            ratios = []
            for op in g.operations:
                if op.actual_attachment == 0:
                    ratios.append(0.)
                elif op.actual_attachment == 1:
                    ratios.append(1.)
                else:
                    ratios.append(self.groups[op.actual_attachment-2].static_data['residual_ratio'])
            g.static_data['residual_ratio'] = np.mean(np.array(ratios))


        indexes = self.get_unused_group_indexes()
        values = [self.groups[i - 2].static_data['residual_ratio'] for i in indexes]
        print(f'residual radio: {np.mean(np.array(values))}')

        # find all paths through the cell
        def list_contains_list(list_of_lists, list_to_check):
            contains = False
            for sub_list in list_of_lists:
                if len(sub_list) != len(list_to_check):
                    continue
                mismatch = False
                for index in range(len(sub_list)):
                    if sub_list[index] != list_to_check[index]:
                        mismatch = True
                        break

                if not mismatch:
                    return True
            return False
        paths = [[0], [1]]
        for g in self.groups:
            for op in g.operations:
                to_add = []
                for path in paths:
                    if path[-1] == op.actual_attachment:
                        new = path.copy()
                        new.append(op.attachment_index)
                        to_add.append(new)
                paths.extend(to_add)
        non_duplicate = []
        for path in paths:
            path_zero = path.copy()
            # path_zero[0] = 0
            if not list_contains_list(non_duplicate, path_zero):
                non_duplicate.append(path_zero)
        final_paths = []
        for path in non_duplicate:
            if path[-1] in indexes:
                final_paths.append(path)
        final_paths.sort(key=lambda x: len(x))


        # find spread data
        path_data = []
        def get_spread(op_type: OperationType):
            if op_type == OperationType.SEP_3X3:
                return 1, 1/9
            elif op_type == OperationType.SEP_5X5:
                return 2, 1/25
            elif op_type == OperationType.SEP_7X7:
                return 3, 1/49
            elif op_type == OperationType.AVG_3X3:
                return 1, 1/9
            elif op_type == OperationType.MAX_3X3:
                return 1, 1/9
            elif op_type == OperationType.DIL_3X3:
                return 2, 1/9
            elif op_type == OperationType.SEP_1X7_7X1:
                return 3, 1/49
            else:
                return 1, 1
        for path in final_paths:
            data = {'spread': 0, 'spread_power': 1}
            for index, group_index in enumerate(path[1:]):
                for op in self.groups[group_index-2].operations:
                    # print(f'path[index-1]: {path[index]}, actual attachment: {op.actual_attachment}')
                    if op.actual_attachment == path[index]:
                        # this means that this op is consuming the connecting node
                        spread, spread_power = get_spread(op.operation_type)
                        data['spread'] += spread
                        data['spread_power'] *= spread_power
            path_data.append(data)
        paths_organized_by_length = {}
        for path_index, path in enumerate(final_paths):
            key = len(path)
            if key not in paths_organized_by_length:
                paths_organized_by_length[key] = []
            paths_organized_by_length[key].append(path_index)
        path_data_organized_by_length = {key:{'spread': 0, 'spread_power': 1} for key, _ in paths_organized_by_length.items()}
        for key, path_indexes in paths_organized_by_length.items():
            for path_index in path_indexes:
                for data_key, data_value in path_data[path_index].items():
                    path_data_organized_by_length[key][data_key] += data_value
            for data_key, data_item in path_data_organized_by_length[key].items():
                path_data_organized_by_length[key][data_key] /= len(path_indexes)

        print(f'data avg: {path_data_organized_by_length}')

        #find critical path
        #find least critical path
        #find average path



class MetaModel(SerialData):
    def __init__(self, hyperparameters: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.cells: List[MetaCell] = []  # [NORMAL_CELL, REDUCTION_CELL]
        self.hyperparameters = hyperparameters

        self.model_name = 'evo_' + str(time.time())
        self.parent_model_name = ''
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
        cell_index, group_index, item_index, mutation_type, mutation_subtype = self.select_mutation()
        self.apply_mutation(cell_index, group_index, item_index, mutation_type, mutation_subtype)

    def select_mutation(self):
        cell_index = int(np.random.random() * len(self.cells))
        select_block = self.cells[cell_index]
        group_index = int(np.random.random() * len(select_block.groups))
        select_group = select_block.groups[group_index]
        item_index = int(np.random.random() * len(select_group.operations))
        mutation_type = np.random.random()
        mutation_subtype = np.random.random()

        return cell_index, group_index, item_index, mutation_type, mutation_subtype

    def apply_mutation(self, cell_index, group_index, item_index, mutation_type, mutation_subtype):
        other_mutation_threshold = ((1. - self.hyperparameters.parameters['IDENTITY_THRESHOLD']) / 2.) + self.hyperparameters.parameters['IDENTITY_THRESHOLD']
        select_block = self.cells[cell_index]
        select_group = select_block.groups[group_index]
        select_item = select_group.operations[item_index]

        mutation_string = f'mutating cell {cell_index}, group {group_index}, item {item_index}: '
        if mutation_type < self.hyperparameters.parameters['IDENTITY_THRESHOLD']:
            # identity mutation
            print(mutation_string + 'identity mutation')
            return

        if self.hyperparameters.parameters['IDENTITY_THRESHOLD'] < mutation_type < other_mutation_threshold:
            # hidden state mutation = change inputs

            # don't try to change the state of the first group since it need to point to the first two inputs of the block
            if group_index != 0:
                previous_attachment = select_item.actual_attachment
                new_attachment = previous_attachment
                # ensure that the mutation doesn't result in the same attachment as before
                while new_attachment == previous_attachment:
                    new_attachment = int(mutation_subtype * select_item.attachment_index) #TODO: EXCLUSIVE RANDOM

                if self.keras_model_data is not None:
                    self.keras_model_data.hidden_state_mutation(self.hyperparameters, cell_index, group_index, item_index, new_attachment, select_item.operation_type)
                select_item.actual_attachment = new_attachment
                print(mutation_string + f'hidden state mutation from {previous_attachment} to {select_item.actual_attachment}')
            else:
                print(mutation_string + f'skipping state mutation for group 0')

        else:
            # operation mutation
            previous_op = select_item.operation_type
            select_item.operation_type = int(mutation_subtype * (OperationType.SEP_1X7_7X1 + 1))
            if previous_op != select_item.operation_type and self.keras_model_data is not None:
                self.keras_model_data.operation_mutation(self.hyperparameters, cell_index, group_index, item_index, select_item.operation_type)
            print(mutation_string + f'operation type mutation from {previous_op} to {select_item.operation_type}')


        initial_layer_shape = self.keras_model.layers[0].get_input_shape_at(0)[1:]

        self.keras_model = None
        self.build_model(initial_layer_shape, False)

    def serialize(self) -> dict:
        return {
            'blocks': [x.serialize() for x in self.cells],
            'metrics': self.metrics.serialize(),
            'hyperparameters': self.hyperparameters.serialize(),
            'model_name': self.model_name,
            'parent_model_name': self.parent_model_name
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
        if 'parent_model_name' in obj:
            self.parent_model_name = obj['parent_model_name']
        else:
            self.parent_model_name = ''

    def build_model(self, input_shape, use_new_weights: bool = True) -> None:
        if self.keras_model is None:
            print('creating model')
            build_time = time.time()
            if self.keras_model_data is None or use_new_weights:
                print('using new data for model')
                self.keras_model_data = ModelDataHolder(self)
            model_input = tf.keras.Input(input_shape)
            self.keras_model = self.keras_model_data.build(model_input)
            build_time = time.time() - build_time
            # optimizer = tf.keras.optimizers.Adam(self.hyperparameters.parameters['MAXIMUM_LEARNING_RATE'])
            optimizer = tf.keras.optimizers.SGD(self.hyperparameters.parameters['MAXIMUM_LEARNING_RATE'])

            compile_time = time.time()
            self.keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            compile_time = time.time() - compile_time

            self.metrics.metrics['build_time'] = build_time
            self.metrics.metrics['compile_time'] = compile_time
        else:
            print('reusing previous keras model')

    def evaluate(self, dataset: ImageDataset) -> None:
        batch_size = self.hyperparameters.parameters['BATCH_SIZE']
        min_lr = self.hyperparameters.parameters['MINIMUM_LEARNING_RATE']
        max_lr = self.hyperparameters.parameters['MAXIMUM_LEARNING_RATE']

        sgdr = SGDR(min_lr, max_lr, batch_size, len(dataset.train_labels),
                    self.hyperparameters.parameters['SGDR_EPOCHS_PER_RESTART'],
                    self.hyperparameters.parameters['SGDR_LR_DECAY'],
                    self.hyperparameters.parameters['SGDR_PERIOD_DECAY'])

        completed_epochs = len(self.metrics.metrics['accuracy'])
        if completed_epochs != 0:
            sgdr.init_after_epochs(completed_epochs)

        callbacks = [self.keras_model_data.drop_path_tracker]
        if self.hyperparameters.parameters['USE_SGDR']:
            callbacks.append(sgdr)

        # print(self.keras_model.summary(line_length=200))
        # print(self.keras_model.non_trainable_variables)

        for iteration in range(int(self.hyperparameters.parameters['TRAIN_ITERATIONS'])):
            print(f'Starting training iteration {iteration}')
            train_time = time.time()
            for epoch_num in range(int(self.hyperparameters.parameters['TRAIN_EPOCHS'])):
                self.keras_model.fit(dataset.train_images, dataset.train_labels, shuffle=True, batch_size=batch_size, epochs=1, callbacks=callbacks)
            train_time = time.time() - train_time

            inference_time = time.time()
            evaluated_metrics = self.keras_model.evaluate(dataset.test_images, dataset.test_labels)
            inference_time = time.time() - inference_time

            self.metrics.metrics['accuracy'].append(float(evaluated_metrics[-1]))
            self.metrics.metrics['average_train_time'].append(train_time / float(self.hyperparameters.parameters['TRAIN_EPOCHS'] * len(dataset.train_labels)))
            self.metrics.metrics['average_inference_time'].append(inference_time / float(len(dataset.test_images)))

    def save_metadata(self, dir_path: str = model_save_dir):
        dir_name = os.path.join(dir_path, self.model_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        SerialData.write_serial_data_to_json(self, dir_name, self.model_name)

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
            custom_objects = {
                'SeperableConvolutionOperation': SeperableConvolutionOperation,
                'AveragePoolingOperation': AveragePoolingOperation,
                'MaxPoolingOperation': MaxPoolingOperation,
                'DoublySeperableConvoutionOperation': DoublySeperableConvoutionOperation,
                'DimensionalityReductionOperation': DimensionalityChangeOperation,
                'IdentityOperation': IdentityOperation,
                'DenseOperation': DenseOperation,
                'Relu6Layer': Relu6Layer
            }

            print(f'saving graph for {self.model_name}')
            dir_name = os.path.join(dir_path, self.model_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            save_time = time.time()


            ModelUtilities.save_keras_model(self.keras_model, dir_name, self.model_name)
            save_time = time.time() - save_time
            self.metrics.metrics['save_time'] = save_time
            print(f'finished saving graph for {self.model_name} after {save_time} seconds')

    def clear_model(self):
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
            if len(fl) > 3 and fl[-3:] == '.h5':
                contains_keras_model = True

        if contains_keras_model:
            print(f'loading model for {self.model_name}')
            load_time = time.time()

            custom_objects = {
                'SeperableConvolutionOperation': SeperableConvolutionOperation,
                'AveragePoolingOperation': AveragePoolingOperation,
                'MaxPoolingOperation': MaxPoolingOperation,
                'DoublySeperableConvoutionOperation': DoublySeperableConvoutionOperation,
                'DimensionalityChangeOperation': DimensionalityChangeOperation,
                'IdentityReductionOperation': IdentityReductionOperation,
                'IdentityOperation': IdentityOperation,
                'DenseOperation': DenseOperation,
                'Relu6Layer': Relu6Layer,
                'DropPathOperation': DropPathOperation
            }
            self.keras_model = ModelUtilities.load_keras_model(dir_name, self.model_name, custom_objects)
            # print(self.keras_model.summary(line_length=200))
            self.keras_model_data = ModelDataHolder(self, self.keras_model)
            load_time = time.time() - load_time
            print(f'finished loading model for {self.model_name} in {load_time} seconds')
            return True
        else:
            print(f'could not find keras model for {self.model_name}')
            return False

    @staticmethod
    def load(dir_path: str, name: str, load_graph: bool = False) -> MetaModel:
        # print(f'loading model, load_graph = {load_graph}')
        dir_name = os.path.join(dir_path, name)
        if not os.path.exists(dir_name):
            print('Model does not exist at specified location')
            return MetaModel()

        serial_data = SerialData.load_serial_data_from_json(dir_name, name)
        result = MetaModel()
        result.deserialize(serial_data)
        if load_graph:
            result.load_model(dir_path)

        return result

    def generate_graph(self, dir_path: str):
        print(f'Generating graph for {self.model_name}')
        graph = Digraph(comment='Model Architecture', format='png')

        for cell_index, cell in enumerate(self.cells):
            graph.node(f'{cell_index}_in', f'Cell Input {cell_index}')
            graph.node(f'{cell_index}_0', f'Previous Layer')
            graph.node(f'{cell_index}_1', f'Residual')
            graph.edge(f'{cell_index}_in', f'{cell_index}_0')
            graph.edge(f'{cell_index}_in', f'{cell_index}_1')
            for group_index, group in enumerate(cell.groups):
                graph.node(f'{cell_index}_{group_index + 2}', f'Group Concat {cell_index}_{group_index}')
                for item_index, item in enumerate(group.operations):
                    graph.node(f'{cell_index}_{group_index}_{item_index}', f'{OperationType.lookup_string(item.operation_type)}')
                    graph.edge(f'{cell_index}_{item.actual_attachment}', f'{cell_index}_{group_index}_{item_index}')
                    graph.edge(f'{cell_index}_{group_index}_{item_index}', f'{cell_index}_{group_index + 2}')

            unused_nodes = cell.get_unused_group_indexes()
            graph.node(f'{cell_index}_out', 'Cell Output')
            for node in unused_nodes:
                graph.edge(f'{cell_index}_{node}', f'{cell_index}_out')

        graph.render(os.path.join(dir_path, self.model_name, 'graph.png'))

    def get_flops(self, dataset:ImageDataset):
        if self.keras_model is None:
            return 0

        # session = tf.compat.v1.get_default_session()
        session = tf.compat.v1.keras.backend.get_session()

        with session.as_default():
            input_img = tf.ones((1,) + dataset.images_shape, dtype=tf.float32)
            output_image = self.keras_model(input_img)

            run_meta = tf.compat.v1.RunMetadata()

            _ = session.run(output_image,
                            options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                            run_metadata=run_meta,
                            # feed_dict={input_img:np.reshape(dataset.test_images[0], (1,)+dataset.images_shape)}
                            )

            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops

        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        # return flops.total_float_ops

    def get_embedding(self):
        embedding = []
        embedding.append(self.hyperparameters.parameters['NORMAL_CELL_N'])
        embedding.append(self.hyperparameters.parameters['CELL_LAYERS'])

        for cell in self.cells:
            for group in cell.groups:
                for op in group.operations:
                    embedding.append(op.operation_type)
                    embedding.append(op.actual_attachment)

        return embedding

    def populate_from_embedding(self, embedding):
        print(f'Populating model from embedding')
        num_cells = 2
        num_groups_per_cell = 5
        num_ops_per_group = 2
        num_cell_inputs = 2

        dup_embedding = embedding.copy()

        self.hyperparameters.parameters['NORMAL_CELL_N'] = dup_embedding[0]
        del dup_embedding[0]
        self.hyperparameters.parameters['CELL_LAYERS'] = dup_embedding[0]
        del dup_embedding[0]

        for cell_ind in range(num_cells):
            self.cells.append(MetaCell(num_cell_inputs))
            for group_ind in range(num_groups_per_cell):
                self.cells[cell_ind].groups.append(MetaGroup())
                for op_ind in range(num_ops_per_group):
                    self.cells[cell_ind].groups[group_ind].operations.append(MetaOperation(num_cell_inputs + group_ind))
                    ref_op = self.cells[cell_ind].groups[group_ind].operations[op_ind]
                    ref_op.operation_type = dup_embedding[0]
                    ref_op.actual_attachment = dup_embedding[1]
                    del dup_embedding[0]
                    del dup_embedding[0]

    def get_confusion_matrix(self, dataset):
        predictions = self.keras_model.predict(dataset.test_images, batch_size=32)

        predictions = ModelUtilities.softmax(predictions)
        predictions = np.argmax(predictions, axis=1)

        matrix = tf.math.confusion_matrix(dataset.test_labels, predictions, num_classes=10)

        matrix_val = None

        if not tf.executing_eagerly():
            with tf.compat.v1.Session().as_default():
                matrix_val = matrix.eval()
        else:
            matrix_val = matrix.numpy()

        return matrix_val

    def activation_viewer(self) -> tf.keras.Model:
        if self.keras_model is None or self.keras_model_data is None:
            return None

        parser = ModelParsingHelper()

        first_cell_reduce = self.keras_model.get_layer(parser.get_next_name('concatenate')).get_output_at(0)
        # first_cell_reduce = tf.keras.layers.Softmax()(first_cell_reduce)

        outputs = [first_cell_reduce]
        outputs.extend(self.keras_model.outputs)
        output_model = tf.keras.Model(inputs=self.keras_model.inputs, outputs=outputs)

        return output_model

    @staticmethod
    def get_nasnet_embedding() -> List:
        return [5, 3,
         OperationType.SEP_3X3, 0,  # NORMAL CELL
         OperationType.IDENTITY, 0,
         OperationType.SEP_3X3, 1,
         OperationType.SEP_5X5, 0,
         OperationType.AVG_3X3, 0,
         OperationType.IDENTITY, 1,
         OperationType.AVG_3X3, 1,
         OperationType.AVG_3X3, 1,
         OperationType.SEP_5X5, 1,
         OperationType.SEP_3X3, 1,
         OperationType.SEP_7X7, 1,  # REDUCTION CELL
         OperationType.SEP_5X5, 0,
         OperationType.MAX_3X3, 0,
         OperationType.SEP_7X7, 1,
         OperationType.AVG_3X3, 0,
         OperationType.SEP_5X5, 1,
         OperationType.MAX_3X3, 0,
         OperationType.SEP_3X3, 2,
         OperationType.AVG_3X3, 2,
         OperationType.IDENTITY, 3]

    @staticmethod
    def get_identity_embedding() -> List:
        embedding = [5, 3]
        embedding.extend([0] * 40)
        return embedding

    @staticmethod
    def get_s1_embedding() -> List:
        return [5, 3, 6, 0, 0, 1, 0, 0, 6, 1, 4, 1, 3, 2, 5, 4, 5, 3, 2, 1, 0, 1, 3, 0, 0, 1, 4, 2, 1, 0, 1, 1, 6, 3, 3, 4, 5, 0, 5, 3, 2, 4]

    @staticmethod
    def get_m1_sep7_embedding() -> List:
        embedding = [5, 3]
        embedding.extend([OperationType.SEP_7X7, 0] * 20)
        return embedding

    @staticmethod
    def get_m1_sep3_embedding() -> List:
        embedding = [5, 3]
        embedding.extend([OperationType.SEP_3X3, 0] * 20)
        return embedding

    @staticmethod
    def get_m1_sep3_serial_embedding() -> List:
        embedding = [5, 3]
        for j in range(2):
            embedding.extend([OperationType.SEP_3X3, 0, OperationType.SEP_3X3, 1])
            for i in range(1, 5):
                embedding.extend([OperationType.SEP_3X3, i + 1, OperationType.SEP_3X3, i + 1])
        return embedding


if __name__ == '__main__':
    pass

