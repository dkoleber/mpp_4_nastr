from __future__ import annotations
import matplotlib.pyplot as plt
from model.MetaModel import *
from Hyperparameters import Hyperparameters
import cv2
import math

from unused.auto_dataset.TaskGen import DatasetGenerator

HERE = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf


def test_accuracy_at_different_train_amounts():
    dir_path = os.path.join(evo_dir, 'test_accuracy_epochs')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    hyperparameters = Hyperparameters()
    hyperparameters.parameters['POPULATION_SIZE'] = 32
    hyperparameters.parameters['ROUNDS'] = 0
    hyperparameters.parameters['TRAIN_EPOCHS'] = 1
    hyperparameters.parameters['TRAIN_ITERATIONS'] = 16

    dataset = ImageDataset.get_cifar10()

    existing_sims = [x for x in os.listdir(dir_path) if 'small' not in x and '.png' not in x]

    num_already_done = len(existing_sims)
    num_remaining = hyperparameters.parameters['POPULATION_SIZE'] - num_already_done
    total_todo = hyperparameters.parameters['POPULATION_SIZE']
    population = []
    for round_num in range(num_remaining):
        print(f'Evaluating model {round_num + 1 + num_already_done} of {total_todo}')
        new_candidate = MetaModel(hyperparameters)
        new_candidate.populate_with_nasnet_metacells()
        new_candidate.model_name = 'evo_' + str(time.time())  # this is redone here since all models are initialized within microseconds of eachother for init population
        new_candidate.build_model(dataset.images_shape)
        new_candidate.evaluate(dataset)
        new_candidate.save_model(dir_path)
        # new_candidate.metrics.metrics['accuracy'].extend([x + round_num for x in range(4)])
        new_candidate.save_metadata(dir_path)
        population.append(new_candidate)
        new_candidate.clear_model()


def convert_all_models():
    dir_path = os.path.join(evo_dir, 'test_accuracy_epochs')
    output_dir_path = os.path.join(evo_dir, 'test_accuracy_epochs_h5')

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # dataset = Dataset.get_cifar10_reduced()
    dataset = ImageDataset.get_cifar10()

    completed_model_names = [x for x in os.listdir(output_dir_path) if '.png' not in x]
    saved_model_names = [x for x in os.listdir(dir_path) if '.png' not in x and x not in completed_model_names]

    for index, model_name in enumerate(saved_model_names):
        print(f'training model {index} of {len(saved_model_names)}')
        model = MetaModel.load(dir_path, model_name, False)

        # model.hyperparameters.parameters['TRAIN_ITERATIONS'] = 1

        model.build_model(dataset.images_shape)
        model.evaluate(dataset)
        model.save_metadata(output_dir_path)
        model.save_model(output_dir_path)
        # model.keras_model.save(os.path.join(output_dir_path, model_name, f'{model_name}.h5'))
        model.clear_model()
        tf.keras.backend.clear_session()


def test_get_flops():
    dir_path = os.path.join(evo_dir, 'test_accuracy_epochs_h5')

    sample = os.listdir(dir_path)[0]

    dataset = ImageDataset.get_cifar10()

    model = MetaModel.load(dir_path, sample, True)
    flops = model.get_flops(dataset)
    print(f'model flops: {flops}')


def train_models_more(dir_name_in, dir_name_out, extra_epochs):
    in_dir_path = os.path.join(evo_dir, dir_name_in)
    out_dir_path = os.path.join(evo_dir, dir_name_out)

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    all_samples = [x for x in os.listdir(in_dir_path) if 'small' not in x and '.csv' not in x]
    done_samples = [x for x in os.listdir(out_dir_path) if 'small' not in x and '.csv' not in x]
    todo_samples = [x for x in all_samples if x not in done_samples]
    print(f'{len(todo_samples)} samples remaining')

    dataset = ImageDataset.get_cifar10()

    for index, sample in enumerate(todo_samples):
        print(f'training sample {index} of {len(todo_samples)}')
        model = MetaModel.load(in_dir_path, sample, True)
        init_iterations = model.hyperparameters.parameters['TRAIN_ITERATIONS']
        model.hyperparameters.parameters['TRAIN_ITERATIONS'] = extra_epochs
        # model.hyperparameters.parameters['USE_SGDR'] = False
        model.evaluate(dataset)
        model.hyperparameters.parameters['TRAIN_ITERATIONS'] += init_iterations
        model.save_model(out_dir_path)
        model.save_metadata(out_dir_path)
        model.clear_model()
        tf.keras.backend.clear_session()


def analyze_model_performances(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    samples = [x for x in os.listdir(dir_path) if 'small' not in x and '.csv' not in x and '.png' not in x]

    candidates = [MetaModel.load(dir_path, x, False) for x in samples]

    def get_stats(items):
        avg = np.average(items)
        std = np.std(items)
        max_val = max(items)
        min_val = min(items)
        print(f'average: {avg}')
        print(f'stdev: {std}')
        print(f'max: {max_val}')
        print(f'min: {min_val}')

    accuracies = [x.metrics.metrics['accuracy'][-1] for x in candidates]

    all_accuracies = [x.metrics.metrics['accuracy'] for x in candidates]
    all_accuracies = np.swapaxes(all_accuracies, 0, 1)
    epochs = [x for x in range(len(all_accuracies))]
    candidate_colors = [x for x in range(len(candidates))]


    plt.subplot(1, 1, 1)
    plt.plot(epochs, all_accuracies)
    plt.title('epoch vs accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.savefig(os.path.join(dir_path, 'learning_curves.png'))

    # print(f'--First 16--')
    # get_stats(accuracies[:16])
    #
    # print(f'--Second 16--')
    # get_stats(accuracies[16:])

    print(f'--All--')
    get_stats(accuracies)


def test_nasnet_model_accuracy(nasnet_path):
    dir_path = os.path.join(evo_dir, nasnet_path)
    # dataset = ImageDataset.get_cifar10_reduced()
    dataset = ImageDataset.get_cifar10()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    hyperparameters = Hyperparameters()

    model = MetaModel(hyperparameters)

    model.populate_from_embedding(MetaModel.get_nasnet_embedding())


    model.build_model(dataset.images_shape)
    model.evaluate(dataset)
    model.save_model(dir_path)
    model.generate_graph(dir_path)
    model.save_metadata(dir_path)
    model.clear_model()

    # new_model = MetaModel.load(dir_path, model.model_name, True)
    # new_model.apply_mutation(1, 0, 1, .99, 1. / float(OperationType.SEP_7X7))
    # new_model.evaluate(dataset)


def view_confusion_matrix():
    dir_path = os.path.join(evo_dir, 'nasnet_arch_test_2')
    dataset = ImageDataset.get_cifar10()

    model = os.listdir(dir_path)[-1]

    model = MetaModel.load(dir_path, model, True)

    print(model.get_confusion_matrix(dataset))


def activations_test():
    # dir_path = os.path.join(evo_dir, 'test_accuracy_epochs_h5_add8_2')
    # model_names = [x for x in os.listdir(dir_path) if '.csv' not in x and 'small' not in x and '.png' not in x]
    dir_path = os.path.join(evo_dir, 'nasnet_arch_test_bigger_shuffled')
    model_name = os.listdir(dir_path)[-1]


    dataset = ImageDataset.get_cifar10()

    model = MetaModel.load(dir_path, model_name, True)
    activations_model = model.activation_viewer()

    images_to_sample = 10000


    print(f'--Predicting test images--')
    predictions = activations_model.predict(dataset.test_images[:images_to_sample,:])
    print(f'--Finished predicting test images--')
    # print(predictions[0].shape)

    vals = predictions[0]

    def norm(x):
        max_val = np.amax(x)
        norm_factor = 1. / max_val
        return x * norm_factor

    def reverse_rms(x):
        return np.sqrt(np.mean((x - 1) ** 2, axis=3))
    def rms(x):
        return np.sqrt(np.mean(x ** 2, axis=3))
    def ms(x):
        return np.mean(x ** 2, axis=3)

    # vals =  rms(vals)
    vals = ms(vals)
    vals = norm(vals)

    features_per_image = vals

    features_sorted_by_class = [[] for x in range(10)]

    for index in range(len(dataset.test_labels[:images_to_sample,:])):
        # print(dataset.test_labels[index, 0])
        # print([len(x) for x in features_sorted_by_class])
        features_sorted_by_class[dataset.test_labels[index, 0]].append(features_per_image[index, :, :])

    features_sorted_by_class = [np.array(x) for x in features_sorted_by_class]

    mean_feature_per_class = [np.mean(x, axis=0) for x in features_sorted_by_class]

    # print(f'mean shape {mean_feature_per_class.shape}')

    mean_feature_per_class = [np.tanh(x) for x in mean_feature_per_class]

    feature_1 = mean_feature_per_class[2]

    scale = 10
    resize = (feature_1.shape[0] * scale, feature_1.shape[1] * scale)
    img = cv2.resize(feature_1, resize, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('predictions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cell_performance_test_1():

    hyperparameters = Hyperparameters()

    dataset = ImageDataset.get_cifar10()

    def get_sorted(images, labels):
        sorted_by_class = [[] for _ in range(10)]
        for index in range(len(images)):
            sorted_by_class[labels[index,0]].append(images[index,:,:])

    sorted_train = get_sorted(dataset.train_images, dataset.train_labels)
    sorted_test = get_sorted(dataset.test_images, dataset.test_labels)

    model = MetaModel(hyperparameters)

    # model.populate_with_nasnet_metacells()
    model.populate_from_embedding(MetaModel.get_nasnet_embedding())

    # model.build_model(dataset.images_shape)
    first_cell = CellDataHolder(3, 3, model.cells[0])

    cell_input = tf.keras.Input(dataset.images_shape)
    cell_output = first_cell.build([cell_input, cell_input])
    cell_model = tf.keras.Model(inputs=cell_input, outputs=cell_output)

    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters.parameters['LEARNING_RATE'])

    def loss(real_image, fake_image, output):
        real_maximize = None # TODO: INNER PROUDCT
        fake_minimize = None

    def train_step(input_image_1, input_image_2):
        with tf.GradientTape() as tape:
            image_1_output = cell_model(input_image_1)
            image_2_output = cell_model(input_image_2)

            total_loss = loss(input_image_1, input_image_2, image_1_output) + loss(input_image_2, input_image_1, image_2_output)

        gradient = tape.gradient(loss, cell_model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, cell_model.trainable_variables))


    #minmize/maximize the determinent of the gram matrix for the positive, and minimize for the negative


def cell_performance_test_2():
    dir_path = os.path.join(evo_dir, 'dataset_gen')

    hyperparameters = Hyperparameters()
    hyperparameters.parameters['TARGET_FILTER_DIMS'] = 64


    mods = [
        # ObjectModifier.SizeModifier,
        # ObjectModifier.PerspectiveModifier,
        # ObjectModifier.RotationModifier,
        # ObjectModifier.ColorModifier
        ]
    DatasetGenerator.build_task_dataset(20000, (32, 32), 10, 4, 2, dir_path, modifiers=mods, max_depth_of_target=1)
    dataset = DatasetGenerator.get_task_dataset(dir_path)

    cell_samples = 50
    num_cells = 1

    steps_per_epoch = math.ceil(len(dataset.train_labels) / hyperparameters.parameters['BATCH_SIZE'])
    total_steps = hyperparameters.parameters['TRAIN_ITERATIONS'] * hyperparameters.parameters['TRAIN_EPOCHS'] * steps_per_epoch

    def test_model():
        metamodel = MetaModel(hyperparameters)
        metamodel.populate_with_nasnet_metacells()

        drop_path_tracker = DropPathTracker(hyperparameters.parameters['DROP_PATH_CHANCE'], 0, total_steps)
        first_cell = CellDataHolder(3, hyperparameters.parameters['TARGET_FILTER_DIMS'], metamodel.cells[0], False, drop_path_tracker, 0.)

        def get_model():
            cell_input = tf.keras.Input(dataset.images_shape)
            cell_output = tf.keras.layers.Conv2D(hyperparameters.parameters['TARGET_FILTER_DIMS'], 1, 1, 'same')(cell_input)
            cell_output = first_cell.build([cell_output, cell_output])
            cell_output = cell_output[0]
            cell_output = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(input_tensor=x, axis=[1, 2]))(cell_output)
            cell_output = tf.keras.layers.Dropout(.5)(cell_output)
            cell_output = tf.keras.layers.Dense(10)(cell_output)
            model = tf.keras.Model(inputs=cell_input, outputs=cell_output)
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters.parameters['MAXIMUM_LEARNING_RATE'])
            model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            return model

        accuracies = []
        for i in range(cell_samples):
            cell_model = get_model()
            cell_model.fit(dataset.train_images, dataset.train_labels, shuffle=True, batch_size=hyperparameters.parameters['BATCH_SIZE'], epochs=1, callbacks=[drop_path_tracker])
            model_accuracies = []
            for test_set_index in range(len(dataset.test_set_images)):
                accuracy = cell_model.evaluate(dataset.test_set_images[test_set_index], dataset.test_set_labels[test_set_index])[-1]
                print(f'{dataset.test_set_names[test_set_index]} test set accuracy: {accuracy}')
                model_accuracies.append(accuracy)
            # accuracy = cell_model.evaluate(dataset.test_images, dataset.test_labels)[-1]
            # accuracies.append(accuracy)
            accuracies.append(model_accuracies)
            tf.keras.backend.clear_session()
            del cell_model

        return accuracies, metamodel.get_embedding()

    all_accuracies = []
    all_embeddings = []

    for x in range(num_cells):
        accuracies, embedding = test_model()
        all_accuracies.append(accuracies)
        all_embeddings.append(embedding)

    all_accuracies = np.array(all_accuracies)
    print(f'all accuracies shape: {all_accuracies.shape}')

    for i in range(len(all_accuracies)):
        print(f'embedding: {all_embeddings[i]}, avgs: {np.mean(all_accuracies[i], axis=0)}')

    output_path = os.path.join(dir_path, 'results.json')
    data = {
        'embeddings': all_embeddings,
        'accuracies': all_accuracies.tolist()
    }
    with open(output_path, 'w+') as fl:
        json.dump(data, fl, indent=4)


def cell_performance_test_2_analysis():
    data_path = os.path.join(evo_dir, 'dataset_gen', 'results.json')

    with open(data_path, 'r') as fl:
        data = json.load(fl)

    all_accuracies = data['accuracies']
    all_embeddings = data['embeddings']


    y = []
    x = [x for x in range(len(all_accuracies[0]))]

    for i in range(len(all_embeddings)):
        frame = 10
        accuracies = np.array(all_accuracies[i])
        avg = np.mean(accuracies, axis=0)
        model_diffs = []
        for step in range(len(x)):
            diff_from_avg = np.squeeze(np.mean(accuracies[step:min(step+frame, len(accuracies))], axis=0) - avg)
            model_diffs.append(diff_from_avg)
        y.append(model_diffs)

    y = np.array(y)
    y = np.reshape(y, (len(x), -1))

    plt.subplot(1, 1, 1)
    plt.plot(x, y)
    plt.title('x v y')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def test_model_accuracy_from_embedding(dir_name, embedding):
    dir_path = os.path.join(evo_dir, dir_name)
    # dataset = ImageDataset.get_cifar10_reduced()
    dataset = ImageDataset.get_cifar10()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    hyperparameters = Hyperparameters()

    model = MetaModel(hyperparameters)

    model.populate_from_embedding(embedding)


    model.build_model(dataset.images_shape)
    model.evaluate(dataset)
    model.save_model(dir_path)
    model.generate_graph(dir_path)
    model.save_metadata(dir_path)
    model.clear_model()


if __name__ == '__main__':
    # test_nasnet_model_accuracy('nasnet_arch_test_sgd_dropout')

    # cell_performance_test_2()
    # cell_performance_test_2_analysis()

    # test_load('nasnet_arch_test_sgd_dropout')
    # view_confusion_matrix()
    # activations_test()
    # train_models_more('nasnet_arch_test', 'nasnet_arch_test_2', 8)
    # train_models_more('nasnet_arch_test_2', 'nasnet_arch_test_3', 8)
    # train_models_more('nasnet_arch_test_sgd', 'nasnet_arch_test_sgd_2', 16)

    # train_models_more('test_accuracy_epochs_h5_add8_2', 'test_accuracy_epochs_h5_64', 16)
    # analyze_model_performances('test_accuracy_epochs_h5')
    # analyze_model_performances('test_accuracy_epochs_h5_add8')
    # analyze_model_performances('test_accuracy_epochs_h5_add8_2')
    test_model_accuracy_from_embedding('serial_3x3', MetaModel.get_m1_sep3_serial_embedding())


