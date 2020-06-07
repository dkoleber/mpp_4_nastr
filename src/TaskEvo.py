from __future__ import annotations
import matplotlib.pyplot as plt

import Dataset
from NASModel import *
from Hyperparameters import Hyperparameters
import cv2
import math

from TaskGen import DatasetGenerator, ObjectModifier, visualize_images
from Utils import *

HERE = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf




def build_cell_model(cell: CellDataHolder, input_shape, num_filters, learning_rate):
    cell_input = tf.keras.Input(input_shape)
    cell_output = tf.keras.layers.Conv2D(num_filters, 1, 1, 'same')(cell_input)
    cell_output = cell.build([cell_output, cell_output])
    cell_output = cell_output[0]
    cell_output = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))(cell_output)
    cell_output = tf.keras.layers.Dropout(.5)(cell_output)
    cell_output = tf.keras.layers.Dense(10)(cell_output)
    model = tf.keras.Model(inputs=cell_input, outputs=cell_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# gets the average accuracy on each test set of the dataset, of each cell in a model
def test_model(metamodel: MetaModel, dataset: DatasetGenerator, cell_samples: int):
    scalar = 4 # want to pretend that the drop path has this times many epochs to complete
    steps_per_epoch = math.ceil(len(dataset.train_labels) / metamodel.hyperparameters.parameters['BATCH_SIZE'])
    total_steps = metamodel.hyperparameters.parameters['TRAIN_ITERATIONS'] * metamodel.hyperparameters.parameters['TRAIN_EPOCHS'] * steps_per_epoch * scalar

    # gets the accuracies for num_cell_samples on each test set of the dataset
    def evaluate_meta_cell(meta_cell: MetaCell, num_cell_samples: int):
        drop_path_tracker = DropPathTracker(metamodel.hyperparameters.parameters['DROP_PATH_CHANCE'], 0, total_steps)
        first_cell = CellDataHolder(3, metamodel.hyperparameters.parameters['TARGET_FILTER_DIMS'], meta_cell, False, drop_path_tracker, 0.)
        # first_cell = CellDataHolder(3, metamodel.hyperparameters.parameters['TARGET_FILTER_DIMS'], meta_cell, False, None, 0.)

        accuracies = []
        # train a distribution of the model to reduce the variation caused by random initialization
        for i in range(num_cell_samples):
            cell_model = build_cell_model(first_cell, dataset.images_shape, metamodel.hyperparameters.parameters['TARGET_FILTER_DIMS'], metamodel.hyperparameters.parameters['MAXIMUM_LEARNING_RATE'])

            # train over mixed dataset
            cell_model.fit(dataset.train_images, dataset.train_labels, shuffle=True, batch_size=metamodel.hyperparameters.parameters['BATCH_SIZE'], epochs=2, callbacks=[drop_path_tracker])
            model_accuracies = []

            # test for specific properties by testing different test sets
            for test_set_index in range(len(dataset.test_set_images)):
                accuracy = cell_model.evaluate(dataset.test_set_images[test_set_index], dataset.test_set_labels[test_set_index])[-1]
                model_accuracies.append(accuracy)
            accuracies.append(model_accuracies)
            tf.keras.backend.clear_session()
            del cell_model

        return accuracies

    cell_accuracies = []
    for cell_index, cell in enumerate(metamodel.cells):
        print(f'Evaluating {cell_index} of {len(metamodel.cells)} cells in metamodel')
        accuracies = evaluate_meta_cell(cell, cell_samples)
        average_accuracies = np.mean(np.array(accuracies), axis=0).tolist()
        cell_accuracies.append(average_accuracies)

    return cell_accuracies


def run_test(dir_name):
    cell_samples = 16
    base_population = 8
    evolved_population = 24

    mods = [
        ObjectModifier.SizeModifier,
        ObjectModifier.PerspectiveModifier,
        ObjectModifier.RotationModifier,
        ObjectModifier.ColorModifier
    ]
    hyperparameters = Hyperparameters()

    dir_path = os.path.join(evo_dir, dir_name)
    results_path = os.path.join(dir_path, 'results.json')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # load dataset, or create a new one if one doesn't exist
    dataset_exists = os.path.exists(dir_path) and 'dataset.npy' in os.listdir(dir_path)
    if not dataset_exists:
        print('Generating dataset')
        DatasetGenerator.build_task_dataset(20000, (32, 32), 10, 4, 2, dir_path, modifiers=mods, max_depth_of_target=1)
    dataset = DatasetGenerator.get_task_dataset(dir_path)

    # load previous test results if they exist
    data = {
        'embeddings':[],
        'accuracies':[]
    }
    if os.path.exists(results_path):
        with open(results_path, 'r') as fl:
            data = json.load(fl)

    def save_data():
        with open(results_path, 'w+') as fl:
            json.dump(data, fl, indent=4)

    def get_average_accuracy(model_index: int, cell_index: int):
        return np.mean(data['accuracies'][model_index][cell_index], axis=0)

    existing_population_size = len(data['embeddings'])
    remaining_base_population = 0 if existing_population_size > base_population else base_population - existing_population_size
    remaining_evolved_population = evolved_population if existing_population_size < base_population else evolved_population - (existing_population_size - base_population)

    print(f'Evaluating {remaining_base_population} base candidates ({base_population - remaining_base_population}/{base_population} done) '
          f'and {remaining_evolved_population} evolved candidates ({evolved_population - remaining_evolved_population}/{evolved_population} done)')

    for i in range(remaining_base_population):
        print(f'Evaluating candidates {i} of {remaining_base_population} base candidates')
        metamodel = MetaModel(hyperparameters)
        metamodel.populate_with_nasnet_metacells()
        accuracies = test_model(metamodel, dataset, cell_samples)
        data['embeddings'].append(metamodel.get_embedding())
        data['accuracies'].append(accuracies)
        save_data()

    performances = [performance(x) for x in data['accuracies']]

    def find_best_indexes():
        best_performances = np.full(performances[0].shape, 1., dtype=np.float32)
        best_indexes = np.zeros(performances[0].shape, dtype=np.int)
        for performance_index, x in enumerate(performances):
            for i, entry in enumerate(x):
                if best_performances[i] > entry:
                    best_performances[i] = entry
                    best_indexes[i] = performance_index

        return best_indexes


    for i in range(remaining_evolved_population):
        print(f'Evaluating candidates {i} of {remaining_evolved_population} evolved candidates')
        best_indexes = find_best_indexes()
        print(f'best indexes: {best_indexes}')
        combined_embeddings = combine_embeddings(data['embeddings'][best_indexes[0]], data['embeddings'][best_indexes[1]])
        mutated_embeddings = mutate_cell_from_embedding(combined_embeddings, 0)
        mutated_embeddings = mutate_cell_from_embedding(mutated_embeddings, 1)
        metamodel = MetaModel(hyperparameters)
        metamodel.populate_from_embedding(mutated_embeddings)
        accuracies = test_model(metamodel, dataset, cell_samples)
        data['embeddings'].append(metamodel.get_embedding())
        data['accuracies'].append(accuracies)
        performances.append(performance(accuracies))
        save_data()


def combine_embeddings(embedding_for_first_cell, embedding_for_second_cell):
    combined = embedding_for_first_cell[:22].copy()
    combined.extend(embedding_for_second_cell[22:].copy())
    return combined


def mutate_cell_from_embedding(embedding, cell_index):
    groups_per_cell = 5
    embeddings_per_group = 4
    embeddings_per_cell = groups_per_cell * embeddings_per_group

    # cell_data = embedding[cell_index*embeddings_per_cell + 2, (cell_index+1)*embeddings_per_cell + 2].copy()
    new_embedding = embedding.copy()

    selection = get_random_int(embeddings_per_cell) + 2 + (embeddings_per_cell * cell_index)# +2 for first two embeddings before cell embeddings

    if selection % 2 == 0: # op type
        new_op = get_random_int(OperationType.SEP_1X7_7X1 + 1)
        print(f'(cell {cell_index}) changing op at {selection} from {new_embedding[selection]} to {new_op}')
        new_embedding[selection] = new_op
    else:
        # maxes out at (20 / 4) + 1 = 6, which is the index for the last cell.
        max_val = int((selection - (embeddings_per_cell * cell_index) - 2) / 4) + 1
        # get_random_int excludes the provided value, so the last cell would be able to pick from 0 - 5
        new_attachment = get_random_int(max_val)
        print(f'(cell {cell_index}) changing attachment at {selection} from {new_embedding[selection]} to {new_attachment}')
        new_embedding[selection] = new_attachment

    return new_embedding


def performance(accuracies):
    values = np.array(accuracies) - 1
    values = np.power(values, 2)
    return np.mean(values, axis=1)


def plot_performances():
    data_path = os.path.join(evo_dir, 'cell_task_evo_full', 'results.json')

    with open(data_path, 'r') as fl:
        data = json.load(fl)

    x = [x for x in range(len(data['accuracies']))]
    performances = np.array([performance(x) for x in data['accuracies']])
    plt.subplot(1, 1, 1)
    plt.plot(x, performances)
    plt.title('x v y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_performance_curve():
    increment = .05
    x = [x * increment for x in range(int(1./increment)+1)]
    x_norm = [[x * increment] for x in range(int(1. / increment) + 1)]
    x_inv = [[1. - (x * increment)] for  x in range(int(1./increment)+1)]
    accuracies = [[x * increment, 1. - (x * increment)] for x in range(int(1./increment)+1)]
    performances = performance(accuracies)

    plt.subplot(1, 1, 1)
    plt.plot(x, performance(accuracies))
    plt.plot(x, performance(x_norm))
    plt.plot(x, performance(x_inv))
    plt.title('x v y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def remove_mutations():
    data_path = os.path.join(evo_dir, 'cell_task_evo_full', 'results.json')

    with open(data_path, 'r') as fl:
        data = json.load(fl)

    data['accuracies'] = data['accuracies'][:8]
    data['embeddings'] = data['embeddings'][:8]

    with open(data_path, 'w+') as fl:
        json.dump(data, fl, indent=4)


def test_identity_performance():
    hyperparameters = Hyperparameters()
    hyperparameters.parameters['TARGET_FILTER_DIMS'] = 32

    dir_path = os.path.join(evo_dir, 'cell_task_evo_full')
    dataset = DatasetGenerator.get_task_dataset(dir_path)

    embedding = [5, 3]
    embedding.extend([0]*40)
    metamodel = MetaModel(hyperparameters)
    metamodel.populate_from_embedding(embedding)
    accuracies = test_model(metamodel, dataset, 16)

    print(accuracies)

# puts together a full model based on the top nst cells in results for a cell based search
def test_nth_in_dir(dir_name, n: int):
    dir_path = os.path.join(evo_dir, dir_name)
    data_path = os.path.join(dir_path, 'results.json')

    with open(data_path, 'r') as fl:
        data = json.load(fl)

    performances = [performance(x) for x in data['accuracies']]

    performances_with_indexes = [(performances[i], data['embeddings'][i]) for i in range(len(performances))]
    num_cells = len(performances[0])  # should be 2
    pwi_per_cell = [performances_with_indexes.copy() for i in range(num_cells)]

    for i in range(num_cells):
        pwi_per_cell[i].sort(key=lambda x: x[0][i])

    selected_embeddings = [x[n][1] for x in pwi_per_cell]


    combined_embeddings = combine_embeddings(selected_embeddings[0], selected_embeddings[1])
    print(combined_embeddings)

    hyperparameters = Hyperparameters()
    hyperparameters.parameters['TRAIN_EPOCHS'] = 2
    hyperparameters.parameters['TRAIN_ITERATIONS'] = 16
    # hyperparameters.parameters['SGDR_EPOCHS_PER_RESTART'] = hyperparameters.parameters['TRAIN_ITERATIONS'] * hyperparameters.parameters['TRAIN_EPOCHS'] #effectively makes SGDR into basic cosine annealing

    dataset = ImageDataset.get_cifar10()

    metamodel = MetaModel(hyperparameters)
    metamodel.populate_from_embedding(combined_embeddings)
    metamodel.build_model(dataset.images_shape)
    metamodel.evaluate(dataset)
    metamodel.save_metadata(dir_path)
    metamodel.save_model(dir_path)
    metamodel.clear_model()


def test_nasnet_cell_performance():
    dir_path = os.path.join(evo_dir, 'cell_task_evo_full')

    dataset = DatasetGenerator.get_task_dataset(dir_path)

    cell_samples = 16

    hyperparameters = Hyperparameters()

    metamodel = MetaModel(hyperparameters)
    metamodel.populate_from_embedding(MetaModel.get_nasnet_embedding())
    accuracies = test_model(metamodel, dataset, cell_samples)

    print(accuracies)
    print(performance(accuracies))


def test_benchmark_models():
    dir_path = os.path.join(evo_dir, 'cell_evo_benchmarks_6')
    results_path = os.path.join(dir_path, 'results.json')
    mods = [
        ObjectModifier.SizeModifier,
        ObjectModifier.PerspectiveModifier,
        ObjectModifier.RotationModifier,
        ObjectModifier.ColorModifier
    ]
    hyperparameters = Hyperparameters()
    cell_samples = 8

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # load dataset, or create a new one if one doesn't exist
    dataset_exists = os.path.exists(dir_path) and 'dataset.npy' in os.listdir(dir_path)
    if not dataset_exists:
        print('Generating dataset')
        # size, dim, num_classes, vertices per object, objects per image,
        DatasetGenerator.build_task_dataset(40000, (16, 16), 10, 4, 10, dir_path, modifiers=mods, max_depth_of_target=1)
    dataset = DatasetGenerator.get_task_dataset(dir_path)

    embeddings = [
        MetaModel.get_nasnet_embedding(),
        MetaModel.get_s1_embedding(),
        MetaModel.get_identity_embedding(),
        MetaModel.get_m1_sep3_embedding(),
        # MetaModel.get_m1_sep7_embedding(),
        MetaModel.get_m1_sep3_serial_embedding(),
    ]

    data = {
        'embeddings':[],
        'accuracies':[]
    }
    if os.path.exists(results_path):
        with open(results_path, 'r') as fl:
            data = json.load(fl)

    def save_data():
        with open(results_path, 'w+') as fl:
            json.dump(data, fl, indent=4)

    for e in embeddings:
        metamodel = MetaModel(hyperparameters)
        metamodel.populate_from_embedding(e)
        accuracies = test_model(metamodel, dataset, cell_samples)
        data['embeddings'].append(metamodel.get_embedding())
        data['accuracies'].append(accuracies)
        save_data()

    performances = [performance(x) for x in data['accuracies']]
    print(performances)


def get_graph():
    dir_path = os.path.join(evo_dir, 'cell_evo_benchmarks')
    hyperparameters = Hyperparameters()
    metamodel = MetaModel(hyperparameters)
    metamodel.populate_from_embedding(MetaModel.get_m1_sep3_serial_embedding())
    metamodel.generate_graph(dir_path)


def get_flops_for_keras_model(model, input_shape):
    session = tf.compat.v1.keras.backend.get_session()

    with session.as_default():
        input_img = tf.ones((1,) + input_shape, dtype=tf.float32)
        output_image = model(input_img)

        run_meta = tf.compat.v1.RunMetadata()

        _ = session.run(output_image,
                        options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                        run_metadata=run_meta)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops


def get_flops_for_cell_models_from_embeddings():
    tf.compat.v1.disable_eager_execution()
    dir_path = os.path.join(evo_dir, 'cell_evo_benchmarks')
    dataset = DatasetGenerator.get_task_dataset(dir_path)
    hyperparameters = Hyperparameters()

    embeddings = [
        MetaModel.get_nasnet_embedding(),
        MetaModel.get_s1_embedding(),
        MetaModel.get_identity_embedding(),
        MetaModel.get_m1_sep3_embedding(),
        MetaModel.get_m1_sep7_embedding()
    ]

    flops = []

    for e in embeddings:
        e_flops = []
        metamodel = MetaModel(hyperparameters)
        metamodel.populate_from_embedding(e)

        steps_per_epoch = math.ceil(len(dataset.train_labels) / metamodel.hyperparameters.parameters['BATCH_SIZE'])
        total_steps = metamodel.hyperparameters.parameters['TRAIN_ITERATIONS'] * metamodel.hyperparameters.parameters['TRAIN_EPOCHS'] * steps_per_epoch
        for meta_cell in metamodel.cells:
            drop_path_tracker = DropPathTracker(metamodel.hyperparameters.parameters['DROP_PATH_CHANCE'], 0, total_steps)
            first_cell = CellDataHolder(3, metamodel.hyperparameters.parameters['TARGET_FILTER_DIMS'], meta_cell, False, drop_path_tracker, 0.)

            cell_model = build_cell_model(first_cell, dataset.images_shape, metamodel.hyperparameters.parameters['TARGET_FILTER_DIMS'], metamodel.hyperparameters.parameters['MAXIMUM_LEARNING_RATE'])
            e_flops.append(get_flops_for_keras_model(cell_model, dataset.images_shape))
            tf.keras.backend.clear_session()
            del cell_model

        flops.append(e_flops)
        print(flops)

    print(flops)

if __name__ == '__main__':
    # test_nasnet_cell_performance()
    # run_test('cell_task_evo_full')
    # test_nth_in_dir('cell_task_evo_full', 0)
    # test_nth_in_dir('cell_task_evo_full', -1)
    # plot_performances()
    # test_performance_curve()
    test_benchmark_models()
    # get_flops_for_cell_models_from_embeddings()
    # get_graph()