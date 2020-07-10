import os
import sys

import scipy
import scipy.stats
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from Utils import list_contains_list
from model.MetaModel import  *


def multi_model_test(dir_name = 'static_analysis_samples', num_models=32, hparams=None, emb_queue=None):
    hyperparameters = Hyperparameters() if hparams is None else hparams

    dataset = ImageDataset.get_cifar10()

    embeddings_queue = [] if emb_queue is None else emb_queue

    dir_path = os.path.join(evo_dir, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    existing_model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
    metamodels = [MetaModel.load(dir_path, x, False) for x in existing_model_names]
    existing_embeddings = [x.get_embedding() for x in metamodels]
    remaining_embedding_queue = [x for x in embeddings_queue if not list_contains_list(existing_embeddings, x)]

    unfinished_model_names = []
    for metamodel in metamodels:
        if len(metamodel.metrics.metrics['accuracy']) < metamodel.hyperparameters.parameters['TRAIN_ITERATIONS']:
            unfinished_model_names.append(metamodel.model_name)

    def eval_model(embedding = None, metamodel = None):
        model = metamodel
        if model is None:
            model = MetaModel(hyperparameters)
            if embedding is None:
                model.populate_with_nasnet_metacells()
            else:
                model.populate_from_embedding(embedding)
            model.build_model(dataset.images_shape)
        model.evaluate(dataset, 1, dir_path)
        model.save_metadata(dir_path)
        model.save_model(dir_path)
        model.generate_graph(dir_path)
        model.clear_model()
        tf.keras.backend.clear_session()

    for index, model_name in enumerate(unfinished_model_names):
        print(f'Evaluating unfinished model {index} of {len(unfinished_model_names)}')
        model = MetaModel.load(dir_path, model_name, True)
        eval_model(metamodel=model)

    for index, embedding in enumerate(remaining_embedding_queue):
        print(f'Evaluating queued model {index} of {len(remaining_embedding_queue)}')
        eval_model(embedding=embedding)


    num_done = len([x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))])
    num_remaining = num_models - num_done

    for count in range(num_remaining):
        print(f'Evaluating non-queued model {count} of {num_remaining}')
        eval_model()


def analyze_slices(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])
    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models])
    num_epochs = len(meta_models[0].metrics.metrics['accuracy'])
    num_samples = len(meta_models)

    zscores = scipy.stats.zscore(accuracies, axis=0)
    mean_zscores = np.array([[np.mean(np.array(x[:i+1])) for i in range(num_epochs)] for x in zscores])

    num_splits = int(math.sqrt(num_samples))






def analyze_stuff(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])
    # meta_models = meta_models[int(len(meta_models)/2):]
    # meta_models = meta_models[int(len(meta_models)*3/4):]
    meta_models = meta_models[1:]

    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']])

    num_samples = accuracies.shape[0]
    num_datapoints_per_sample = accuracies.shape[1]

    x = [x for x in range(num_datapoints_per_sample)]

    zscores = scipy.stats.zscore(accuracies, axis=0)

    def se(val1, val2):
        return ((val1 - val2)**2)


    # avg_z_score = np.mean(zscores, axis=1)
    # coorelation = np.corrcoef(avg_z_score, zscores[:, -1])
    # print(f'avg z score coorelation with final z score: {coorelation[0,1]}')
    z_mean_over_time = np.array([[np.mean(np.array(x[:i+1])) for i in range(num_datapoints_per_sample)] for x in zscores])
    correlations_over_time_mean = np.array([np.corrcoef(z_mean_over_time[:, i], zscores[:, -1])[0,1] for i in range(num_datapoints_per_sample)])
    correlations_over_time = np.array([np.corrcoef(zscores[:, i], zscores[:, -1])[0,1] for i in range(num_datapoints_per_sample)])
    z_mean_over_time_error = np.mean(np.array([[se(y, zscores[index][-1]) for y in x] for index, x in enumerate(z_mean_over_time)]),axis=0)
    zscores = np.swapaxes(zscores, 0, 1)
    accuracies = np.swapaxes(accuracies, 0, 1)
    z_mean_over_time = np.swapaxes(z_mean_over_time, 0, 1)
    print(z_mean_over_time_error.shape)
    # z_mean_over_time_error = np.swapaxes(z_mean_over_time_error, 0, 1)

    plt.subplot(4, 1, 1)
    plt.plot(x, zscores)
    plt.title('x v zscores')
    plt.xlabel('x')
    plt.ylabel('zscore')

    plt.subplot(4, 1, 2)
    plt.plot(x, z_mean_over_time)
    plt.title('x v zscore mean up to this x')
    plt.xlabel('x')
    plt.ylabel('zscore')

    plt.subplot(4, 1, 3)
    plt.plot(x, correlations_over_time_mean)
    plt.title('x v zscore mean up to this x correlation with final z score')
    plt.xlabel('x')
    plt.ylabel('correlation')

    plt.subplot(4, 1, 4)
    plt.plot(x, z_mean_over_time_error)
    plt.title('x v mean of zscore mean up to this point MSE with final z score')
    plt.xlabel('x')
    # plt.yscale('log')
    plt.ylabel('mse')

    plt.show()

    ranked_models = [(x.model_name, x.metrics.metrics['accuracy'][-1]) for x in meta_models]
    ranked_models.sort(key=lambda x: x[1])
    print(ranked_models)


def multi_config_test():

    num_models = 16

    def default_params(epochs:int) -> Hyperparameters:
        params = Hyperparameters()
        params.parameters['REDUCTION_EXPANSION_FACTOR'] = 2
        params.parameters['SGDR_EPOCHS_PER_RESTART'] = epochs
        params.parameters['TRAIN_ITERATIONS'] = epochs
        params.parameters['MAXIMUM_LEARNING_RATE'] = 0.025
        params.parameters['MINIMUM_LEARNING_RATE'] = 0.001
        params.parameters['DROP_PATH_TOTAL_STEPS_MULTI'] = 1
        params.parameters['BATCH_SIZE'] = 16
        return params

    def medium_params(epochs:int) -> Hyperparameters:
        params = default_params(epochs)
        params.parameters['TARGET_FILTER_DIMS'] = 32
        params.parameters['NORMAL_CELL_N'] = 6
        params.parameters['CELL_LAYERS'] = 3
        return params

    def small_params(epochs:int) -> Hyperparameters:
        params = default_params(epochs)
        params.parameters['TARGET_FILTER_DIMS'] = 24
        params.parameters['NORMAL_CELL_N'] = 3
        params.parameters['CELL_LAYERS'] = 3
        return params

    def tiny_params(epochs:int) -> Hyperparameters:
        params = default_params(epochs)
        params.parameters['TARGET_FILTER_DIMS'] = 16
        params.parameters['NORMAL_CELL_N'] = 2
        params.parameters['CELL_LAYERS'] = 3
        return params

    # def default_alt_params() -> Hyperparameters:
    #     params = Hyperparameters()
    #     params.parameters['REDUCTION_EXPANSION_FACTOR'] = 2
    #     params.parameters['SGDR_EPOCHS_PER_RESTART'] = 30
    #     params.parameters['TRAIN_ITERATIONS'] = 30
    #     params.parameters['MAXIMUM_LEARNING_RATE'] = 0.025
    #     params.parameters['MINIMUM_LEARNING_RATE'] = 0.001
    #     params.parameters['DROP_PATH_TOTAL_STEPS_MULTI'] = 1
    #     params.parameters['BATCH_SIZE'] = 16
    #     return params
    #
    # def medium_alt_params() -> Hyperparameters:
    #     params = default_params()
    #     params.parameters['TARGET_FILTER_DIMS'] = 36
    #     params.parameters['NORMAL_CELL_N'] = 6
    #     params.parameters['CELL_LAYERS'] = 3
    #     return params
    #
    #
    # def tiny_alt_params() -> Hyperparameters:
    #     params = default_params()
    #     params.parameters['TARGET_FILTER_DIMS'] = 8
    #     params.parameters['NORMAL_CELL_N'] = 6
    #     params.parameters['CELL_LAYERS'] = 3
    #     return params

    embeddings = []
    np.random.seed(0)

    for i in range(num_models):
        m = MetaModel(default_params(0))
        m.populate_with_nasnet_metacells()
        embeddings.append(m.get_embedding())


    multi_model_test('zs_small', num_models=num_models, hparams=small_params(32), emb_queue=embeddings)
    multi_model_test('zs_medium', num_models=num_models, hparams=medium_params(32), emb_queue=embeddings)
    # multi_model_test('zs_tiny', num_models=num_models, hparams=tiny_params(32), emb_queue=embeddings)
    multi_model_test('zs_small_16', num_models=num_models, hparams=small_params(16), emb_queue=embeddings)
    multi_model_test('zs_medium_16', num_models=num_models, hparams=medium_params(16), emb_queue=embeddings)
    # multi_model_test('zs_tiny_16', num_models=num_models, hparams=tiny_params(16), emb_queue=embeddings)
    # multi_model_test('zs_small_48', num_models=num_models, hparams=small_params(48), emb_queue=embeddings)
    # multi_model_test('zs_medium_48', num_models=num_models, hparams=medium_params(48), emb_queue=embeddings)
    # multi_model_test('zs_tiny_48', num_models=num_models, hparams=tiny_params(48), emb_queue=embeddings)

def verify_mutations():
    params = Hyperparameters()
    params.parameters['TRAIN_ITERATIONS'] = 1
    params.parameters['REDUCTION_EXPANSION_FACTOR'] = 2

    dataset = ImageDataset.get_cifar10_reduced()



    for i in range(50):
        model = MetaModel(params)
        model.populate_with_nasnet_metacells()
        model.build_model(dataset.images_shape)
        # model.evaluate(dataset)

        model.mutate()
        model.mutate()
        model.mutate()
        model.clear_model()
        tf.keras.backend.clear_session()
        # model.mutate()
        # model.mutate()
        # model.mutate()
        # model.mutate()
        # model.mutate()
        # model.mutate()
        # model.mutate()

def verify_load():
    dir_path = os.path.join(evo_dir, 'test_load_v2')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    params = Hyperparameters()
    params.parameters['TRAIN_ITERATIONS'] = 1
    params.parameters['REDUCTION_EXPANSION_FACTOR'] = 2

    dataset = ImageDataset.get_cifar10_reduced()

    for i in range(50):
        model = MetaModel(params)
        model.populate_with_nasnet_metacells()
        model.build_model(dataset.images_shape)
        model.save_model(dir_path)
        model.save_metadata(dir_path)
        model.clear_model()
        tf.keras.backend.clear_session()

        other_model = MetaModel.load(dir_path, model.model_name, True)

        tf.keras.backend.clear_session()

if __name__ == '__main__':
    multi_config_test()
    # analyze_stuff('zs_set_1\\zs_medium')
    # analyze_stuff('zs_small')
    # verify_mutations()
    # verify_load()