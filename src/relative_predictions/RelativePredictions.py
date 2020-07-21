import os
import sys

import scipy
import scipy.stats
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from Utils import list_contains_list
from model.MetaModel import  *


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


def se(val1, val2):
    return ((val1 - val2)**2)


def ae(val1, val2):
    return abs(val1 - val2)


def spearman_coef_distinct(rankings_predicted, rankings_actual):
    n = len(rankings_predicted)
    if n < 2:
        return 1.

    top = 6 * sum([(rankings_predicted[i] - rankings_actual[i])**2 for i in range(n)])
    bottom = n * (n**2 - 1)

    return 1. - (top/bottom)

def spearman_coef(rankings_predicted, rankings_actual):
    n = len(rankings_predicted)
    if n < 2:
        return 1.

    cov = np.cov(rankings_predicted, rankings_actual)[0][0]
    std_pred = np.std(rankings_predicted)
    std_act = np.std(rankings_actual)

    if std_pred == 0 or std_act == 0:
        return 1.

    return cov / (std_pred*std_act)

def analyze_slices(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])

    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models])
    num_epochs = len(meta_models[0].metrics.metrics['accuracy'])
    num_models = len(meta_models)

    window_sizes = [num_epochs, int(num_epochs/2), int(num_epochs/3), int(num_epochs/4), int(num_epochs/8), int(num_epochs/16)]

    zscores = scipy.stats.zscore(accuracies, axis=0)
    mean_zscores = np.array([np.array([[np.mean(np.array(x[max(0,i-window_size):i+1])) for i in range(num_epochs)] for x in zscores]) for window_size in window_sizes])


    correlations_over_time = np.array([np.corrcoef(zscores[:, i], zscores[:, -1])[0,1] for i in range(num_epochs)])
    correlations_over_time_mean = np.array([np.array([np.corrcoef(mean_zscores[j][:, i], zscores[:, -1])[0, 1] for i in range(num_epochs)]) for j in range(len(window_sizes))])

    mean_zscore_error = np.array([np.mean(np.array([[se(y, zscores[index][-1]) for y in x] for index, x in enumerate(mean_zscores[j])]), axis=0) for j in range(len(window_sizes))])


    num_simulations = 32
    indexes = np.array([np.arange(num_models) for _ in range(num_simulations)])
    np.random.seed(0)
    for index_set in indexes:
        np.random.shuffle(index_set)

    def predict_with_window(window, prediction_epoch):
        results = []

        for simulation in indexes:
            # print(f'simulating: {simulation}')
            results.append([[],[],[]])
            for num_models_so_far, master_model_index in enumerate(simulation):
                model_accuracies_up_to_point_final = [accuracies[i][-1] for i in simulation[:num_models_so_far + 1]]
                model_zscores_final = scipy.stats.zscore(model_accuracies_up_to_point_final, axis=0) if num_models_so_far > 1 else [0]
                model_accuracies_up_to_point_at_epoch = [accuracies[i][:prediction_epoch + 1] for i in simulation[:num_models_so_far + 1]]
                model_zscores_at_epoch = np.array(scipy.stats.zscore(model_accuracies_up_to_point_at_epoch, axis=0) if num_models_so_far > 1 else [[0]*(prediction_epoch+1)])

                zscore_predictions = np.mean(model_zscores_at_epoch[:,max(0,prediction_epoch+1-window):prediction_epoch+1], axis=1)
                zscore_prediction = zscore_predictions[-1]
                zscore_squared_error = ae(zscore_prediction, model_zscores_final[-1])

                ranks_prediction = np.argsort(zscore_predictions)
                ranks_actual = np.argsort(model_zscores_final)

                average_rank_distance = np.mean([abs(np.where(ranks_prediction==i)[0] - np.where(ranks_actual==i)[0]) for i in range(num_models_so_far+1)]) if num_models_so_far > 1 else 0

                this_rank_distance = abs(np.where(ranks_prediction==num_models_so_far)[0] - np.where(ranks_actual==num_models_so_far)[0]) if num_models_so_far > 1 else 0

                results[-1][0].append(zscore_squared_error)
                results[-1][1].append(average_rank_distance)
                results[-1][2].append(this_rank_distance)

        results = np.array(results)

        # print(results.shape)
        results = np.mean(results, axis=0)

        return results


    def predict_with_windows(prediction_epoch):
        results = np.array([predict_with_window(window, prediction_epoch) for window in window_sizes])
        results = np.swapaxes(results, 0, 2)

        errors = results[:, 0, :]
        ranks_total_distance = results[:, 1, :]
        rank_guess_distance = results[:, 2, :]



        return errors, ranks_total_distance, rank_guess_distance

    r_2_e, r_2_rt, r_2_rg = predict_with_windows(int(num_epochs/2)-1)
    r_3_e, r_3_rt, r_3_rg = predict_with_windows(int(num_epochs/3)-1)
    r_4_e, r_4_rt, r_4_rg = predict_with_windows(int(num_epochs/4)-1)


    zscores = np.swapaxes(zscores, 0, 1)
    mean_zscores = np.swapaxes(mean_zscores, 0, 2)
    mean_zscores = np.reshape(mean_zscores, (num_epochs, -1))
    correlations_over_time_mean = np.swapaxes(correlations_over_time_mean, 0, 1)
    mean_zscore_error = np.swapaxes(mean_zscore_error, 0, 1)

    x = [i for i in range(num_epochs)]
    models_x = [i for i in range(num_models)]
    rows = 5
    cols = 4


    # plt.subplot(rows, 1, 1)
    # plt.title('epoch vs zscore')
    # plt.plot(x, zscores)

    # plt.subplot(num_plots, 1, plot_ind)
    # plot_ind += 1
    # plt.title('epoch vs zscore mean')
    # plt.plot(x, mean_zscores)

    plt.subplot(rows, cols, 1)
    plt.title('epoch vs correlation')
    plt.plot(x, correlations_over_time)

    plt.subplot(rows, cols, 5)
    plt.title('epoch vs correlation of means')
    plt.plot(x, correlations_over_time_mean)

    plt.subplot(rows, cols, 9)
    plt.title('epoch vs mean zscore error')
    plt.plot(x, mean_zscore_error)



    plt.subplot(rows, cols, 2)
    plt.title('error - predict at 1/2')
    plt.plot(models_x, r_2_e)

    plt.subplot(rows, cols, 6)
    plt.title('error - predict at 1/3')
    plt.plot(models_x, r_3_e)

    plt.subplot(rows, cols, 10)
    plt.title('error - predict at 1/4')
    plt.plot(models_x, r_4_e)



    plt.subplot(rows, cols, 3)
    plt.title('rank - predict at 1/2')
    plt.plot(models_x, r_2_rt)

    plt.subplot(rows, cols, 7)
    plt.title('rank - predict at 1/3')
    plt.plot(models_x, r_3_rt)

    plt.subplot(rows, cols, 11)
    plt.title('rank - predict at 1/4')
    plt.plot(models_x, r_4_rt)


    plt.subplot(rows, cols, 4)
    plt.title('rank - predict at 1/2')
    plt.plot(models_x, r_2_rg)

    plt.subplot(rows, cols, 8)
    plt.title('rank - predict at 1/3')
    plt.plot(models_x, r_3_rg)

    plt.subplot(rows, cols, 12)
    plt.title('rank - predict at 1/4')
    plt.plot(models_x, r_4_rg)


    plt.show()

    '''
    blue
    orange
    green
    red
    purple
    brown
    pink
    gray
    yellow
    cyan
    '''


def analyze_stuff(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])
    meta_models = meta_models[int(len(meta_models)/2):]
    # meta_models = meta_models[int(len(meta_models)*3/4):]
    # meta_models = meta_models[1:]

    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']])

    num_samples = accuracies.shape[0]
    num_datapoints_per_sample = accuracies.shape[1]

    x = [x for x in range(num_datapoints_per_sample)]

    zscores = scipy.stats.zscore(accuracies, axis=0)

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


def update_hparams(dir_name):
    dir_path = os.path.join(evo_dir, dir_name)
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]

    for x in meta_models:
        x.save_metadata(dir_path)


def analyze_multiple():
    def load_accuracies(dir_name):

        dir_path = os.path.join(evo_dir, dir_name)
        model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

        meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
        meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]
        meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])

        accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models])

        return accuracies


    dir_names = ['zs_standard_6x3_32e_32f', 'zs_small_3x3_32e_24f', 'zs_medium_5x3_16e_32f', 'zs_medium_5x3_16e_24f']

    accuracies = [load_accuracies(x) for x in dir_names]

    num_experiments = len(dir_names)
    num_epochs = [len(x[0]) for x in accuracies]
    num_models = len(accuracies[0])

    # window_size_scalars = [1, .5, 1/3, 1/4, 1/8, 1/16]
    # zscores = [scipy.stats.zscore(x, axis=0) for x in accuracies]
    # def zscore_means_over_window(window_scalar):
    #     actual_windows = [int(window_scalar * x) for x in num_epochs]
    #     return [[[np.mean(np.array(x[max(0, i - actual_windows[exp_ind]):i + 1])) for i in range(num_epochs[exp_ind])] for x in exp] for exp_ind, exp in enumerate(zscores)]
    # zscore_means_over_windows = [zscore_means_over_window(window) for window in window_size_scalars]
    # zscore_means_over_window_errors = [[np.mean([[ae(epoch, zscores[exp_ind][sample_ind][-1]) for epoch in sample] for sample_ind, sample in enumerate(exp)]) for exp_ind, exp in enumerate(window)] for window in zscore_means_over_windows]

    def predict_with_window(accs, window_scalar, prediction_epoch_scalar):
        num_epochs_for_this_exp = len(accs[0])
        prediction_epoch = int(num_epochs_for_this_exp * prediction_epoch_scalar)
        # window = max(int(num_epochs_for_this_exp * window_scalar),1)
        window = max(int(prediction_epoch * window_scalar),1)

        results = []
        num_simulations = 32
        indexes = np.array([np.arange(num_models) for _ in range(num_simulations)])
        np.random.seed(0)
        for index_set in indexes:
            np.random.shuffle(index_set)

        for simulation in indexes:
            # print(f'simulating: {simulation}')
            results.append([[],[],[],[]])
            for num_models_so_far, master_model_index in enumerate(simulation):

                model_accuracies_up_to_point_final = [accs[i][-1] for i in simulation[:num_models_so_far + 1]]
                model_zscores_final = scipy.stats.zscore(model_accuracies_up_to_point_final, axis=0) if num_models_so_far >= 1 else [0]

                model_accuracies_up_to_point_at_epoch = np.array([accs[i][:prediction_epoch] for i in simulation[:num_models_so_far + 1]])
                # std = np.std(model_accuracies_up_to_point_at_epoch, axis=0)
                # print(f'e: {num_models_so_far}, pe: {prediction_epoch} s1: {model_accuracies_up_to_point_at_epoch.shape} s: {std.shape}')
                model_zscores_at_epoch = np.array(scipy.stats.zscore(model_accuracies_up_to_point_at_epoch, axis=0) if num_models_so_far >= 1 else [[0]*(prediction_epoch)])

                zscore_predictions = np.mean(model_zscores_at_epoch[:,max(0,prediction_epoch-window):prediction_epoch], axis=1)
                zscore_prediction = zscore_predictions[-1]
                zscore_error = ae(zscore_prediction, model_zscores_final[-1])

                ranks_prediction = np.argsort(zscore_predictions)
                ranks_actual = np.argsort(model_zscores_final)

                spearman_distinct = spearman_coef_distinct(ranks_prediction, ranks_actual)

                average_rank_distance = np.mean([abs(np.where(ranks_prediction==i)[0] - np.where(ranks_actual==i)[0]) for i in range(num_models_so_far+1)]) if num_models_so_far >= 1 else 0

                this_rank_distance = abs(np.where(ranks_prediction==num_models_so_far)[0] - np.where(ranks_actual==num_models_so_far)[0]) if num_models_so_far >= 1 else 0

                results[-1][0].append(zscore_error)
                results[-1][1].append(average_rank_distance)
                results[-1][2].append(this_rank_distance)
                results[-1][3].append(spearman_distinct)

        results = np.array(results)

        # print(results.shape)
        results = np.mean(results, axis=0)

        return results

    chosen_windows = [1, .5, .25, .001]
    chosen_prediction_scalars = [1, .5, .25, .125]

    x_models = [x for x in range(num_models)]
    rows = len(chosen_windows)

    for pred in chosen_prediction_scalars:
        acceleration = 1/pred
        name = f'{acceleration}x acceleration'
        plt.figure(num=name,figsize=(16,14))
        this_prediction_scalar = []
        num_plots = 0
        for window in chosen_windows:
            val = np.array([predict_with_window(acc, window, pred) for acc in accuracies])

            this_prediction_scalar.append(val)

            num_datapoints_per_sim = len(val[0])
            cols = num_datapoints_per_sim

            point_names = ['prediction error', 'avg rank distance', 'this rank distance', 'spearman coef distinct', 'spearman coef real']
            for datapoint in range(num_datapoints_per_sim):
                num_plots += 1
                plt.subplot(rows, cols, num_plots)
                datapoint_rearranged = np.swapaxes(val[:,datapoint,:], 0, 1)
                plt.title(f'{window} window, {point_names[datapoint]}')
                plt.plot(x_models, datapoint_rearranged)

        # plt.show()
        plt.savefig(f'{name}.jpg')


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
        params.parameters['CELL_STACKS'] = [6, 1]
        params.parameters['CELL_LAYERS'] = 3
        return params

    def small_params(epochs:int) -> Hyperparameters:
        params = default_params(epochs)
        params.parameters['TARGET_FILTER_DIMS'] = 24
        params.parameters['CELL_STACKS'] = [3, 1]
        params.parameters['CELL_LAYERS'] = 3
        return params


    def long_params() -> Hyperparameters:
        params = default_params(16)
        params.parameters['TARGET_FILTER_DIMS'] = 16
        params.parameters['CELL_STACKS'] = [3, 1]
        params.parameters['CELL_LAYERS'] = 2
        params.parameters['CONCATENATE_ALL'] = False
        params.parameters['GROUPS_PER_CELL'] = 7
        return params


    embeddings = []
    np.random.seed(0)

    for i in range(num_models):
        m = MetaModel(default_params(0))
        m.populate_with_nasnet_metacells()
        embeddings.append(m.get_embedding())

    np.random.seed(0)
    long_embeddings = []
    for i in range(num_models):
        m = MetaModel(long_params())
        m.populate_with_nasnet_metacells()
        long_embeddings.append(m.get_embedding())

    # multi_model_test('zs_small', num_models=num_models, hparams=small_params(32), emb_queue=embeddings)
    # multi_model_test('zs_medium', num_models=num_models, hparams=medium_params(32), emb_queue=embeddings)
    # multi_model_test('zs_tiny', num_models=num_models, hparams=tiny_params(32), emb_queue=embeddings)
    multi_model_test('zs_standard_6x3_32e_32f', num_models=num_models, hparams=medium_params(32), emb_queue=embeddings)
    multi_model_test('zs_medium_6x3_16e_32f', num_models=num_models, hparams=medium_params(16), emb_queue=embeddings)
    multi_model_test('zs_small_3x3_16e_24f', num_models=num_models, hparams=small_params(16), emb_queue=embeddings)
    multi_model_test('zs_long_16', num_models=num_models, hparams=long_params(), emb_queue=long_embeddings)


if __name__ == '__main__':
    analyze_multiple()
    # multi_config_test()

    # analyze_stuff('zs_set_1\\zs_medium')
    # analyze_stuff('zs_small')
    # analyze_stuff('zs_medium')
    # analyze_slices('zs_small')
    # analyze_slices('zs_medium')
    # analyze_slices('zs_set_1\\zs_medium')

