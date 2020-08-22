import os
import sys

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from Utils import list_contains_list
from model.MetaModel import  *

from nas_201_api import NASBench201API as nasapi



def random_exclusive(max_val, n):
    all_vals = np.arange(max_val)
    np.random.shuffle(all_vals)
    return all_vals[:n]


def se(val1, val2):
    return ((val1 - val2)**2)


def ae(val1, val2):
    return abs(val1 - val2)


def get_ranks(measurements):
    return np.argsort(np.argsort(measurements, axis=0), axis=0)


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


def _analyze_multiple(accuracies, num_simulations = 32, prefix=''):

    num_experiments = len(accuracies)
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
        indexes = np.array([np.arange(num_models) for _ in range(num_simulations)])
        np.random.seed(0)
        for index_set in indexes:
            np.random.shuffle(index_set)


        rank_corrs = [[],[]]
        for sim_ind, simulation in enumerate(indexes):
            print(f'simulating: {sim_ind} of {num_simulations}')
            results.append([[],[],[],[],[],[]])
            ranks = [[], []]
            rank_distances = [[], []]
            for num_models_so_far, master_model_index in enumerate(simulation):
                final_accuracies = [accs[i][-1] for i in simulation[:num_models_so_far + 1]]

                def get_rank_stats(ranks_predicted, ranks_actual):
                    spearman = spearman_coef_distinct(ranks_predicted, ranks_actual)
                    average_rank_distance = np.mean([abs(np.where(ranks_predicted == i)[0] - np.where(ranks_actual == i)[0]) for i in range(num_models_so_far + 1)]) if num_models_so_far >= 1 else 0
                    this_rank_prediction = ranks_predicted[-1] # np.where(ranks_predicted == num_models_so_far)[0][0]
                    this_rank_actual = ranks_actual[-1] #np.where(ranks_actual == num_models_so_far)[0][0]
                    this_rank_distance = abs(this_rank_prediction - this_rank_actual) if num_models_so_far >= 1 else 0
                    return spearman, average_rank_distance, this_rank_prediction, this_rank_distance

                accuracies_up_to_prediction_epoch = np.array([accs[i][:prediction_epoch] for i in simulation[:num_models_so_far + 1]])
                final_ranks_actual = get_ranks(final_accuracies)

                # zm measurements
                zscores_up_to_prediction_epoch = np.array(scipy.stats.zscore(accuracies_up_to_prediction_epoch, axis=0) if num_models_so_far >= 1 else [[0]*(prediction_epoch)])
                zscore_predictions = np.mean(zscores_up_to_prediction_epoch[:,max(0,prediction_epoch-window):], axis=1)
                zscore_ranks_prediction = get_ranks(zscore_predictions)
                zscore_spearman, average_zm_rank_distance, this_zm_rank_prediction, this_zm_rank_distance = get_rank_stats(zscore_ranks_prediction, final_ranks_actual)
                ranks[0].append(this_zm_rank_prediction)
                rank_distances[0].append(this_zm_rank_distance)

                # rm measurements
                ranks_up_to_prediction_epoch = get_ranks(accuracies_up_to_prediction_epoch)
                rm_rank_predictions = get_ranks(np.mean(ranks_up_to_prediction_epoch[:,max(0,prediction_epoch-window):], axis=1))
                rm_spearman, average_rm_rank_distance, this_rm_rank_prediction, this_rm_rank_distance = get_rank_stats(rm_rank_predictions, final_ranks_actual)
                ranks[1].append(this_rm_rank_prediction)
                rank_distances[1].append(this_rm_rank_distance)

                results[-1][0].append(average_zm_rank_distance/(num_models_so_far+1))
                results[-1][1].append(this_zm_rank_distance/(num_models_so_far+1))
                results[-1][2].append(zscore_spearman)
                results[-1][3].append(average_rm_rank_distance/(num_models_so_far+1))
                results[-1][4].append(this_rm_rank_distance/(num_models_so_far+1))
                results[-1][5].append(rm_spearman)

                # print(f'{average_zm_rank_distance} {this_rm_rank_distance} {zscore_spearman} | {ranks_up_to_prediction_epoch.shape} {rm_rank_predictions} {average_rm_rank_distance} {this_rm_rank_distance} {rm_spearman}')

            for i in range(len(ranks)):
                ranks_set = ranks[i]
                rank_distances_set = rank_distances[i]
                rank_corr_with_distance = np.corrcoef(ranks_set, rank_distances_set)[0][1] if max(rank_distances_set) != min(rank_distances_set) else 0.
                rank_corrs[i].append(rank_corr_with_distance)


        results = np.array(results)

        results = np.mean(results, axis=0)
        corrs = np.mean(rank_corrs, axis=1)
        # print(corrs.shape)
        # print(np.array(rank_corrs).shape)

        return results, corrs

    chosen_windows = [1, .5, .25, .001]
    chosen_prediction_scalars = [1, .5, .25, .125]

    all_results = []
    for pred in chosen_prediction_scalars:
        pred_results = []
        for window_ind, window in enumerate(chosen_windows):
            results = [predict_with_window(acc, window, pred) for acc in accuracies]
            pred_results.append(results)
        all_results.append(pred_results)



    point_names = ['Average Rank Error', 'New Rank Error', 'Spearman Coef']
    y_ticks = [(0, .5, .1), (0, .5, .1), (0., 1., .2)]
    x_ticks = (0, num_models, int(num_models/4))

    x_models = [x for x in range(num_models)]
    rows = len(chosen_windows)
    rank_corrs = []
    actual_width = 5.5
    height = actual_width

    for pred_ind, pred in enumerate(chosen_prediction_scalars):
        acceleration = 1/pred
        name = f'{prefix}{acceleration}x_acceleration'

        fig = plt.figure(num=name,figsize=(actual_width,height))
        # fig.suptitle(f'{pred} Prediction Scalar')
        this_prediction_scalar = []
        num_plots = 0
        this_window_rank_corrs = []
        for window_ind, window in enumerate(chosen_windows):
            results = all_results[pred_ind][window_ind] #[predict_with_window(acc, window, pred) for acc in accuracies]

            val = np.array([x[0] for x in results])
            this_window_rank_corrs.append([x[1] for x in results])

            this_prediction_scalar.append(val)

            num_datapoints_per_sim = len(val[0])
            num_datapoints_per_category = int(num_datapoints_per_sim / 2)
            cols = num_datapoints_per_category

            for datapoint in range(num_datapoints_per_category):
                num_plots += 1
                ax = plt.subplot(rows, cols, num_plots)
                if datapoint == 0:
                    ax.set_ylabel(f'{window} window')
                if window_ind == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel(point_names[datapoint])

                plt.xticks(np.arange(x_ticks[0], x_ticks[1], x_ticks[2]))
                plt.xlim((x_ticks[0], x_ticks[1]))
                plt.yticks(np.arange(y_ticks[datapoint][0], y_ticks[datapoint][1], y_ticks[datapoint][2]))
                plt.ylim((y_ticks[datapoint][0], y_ticks[datapoint][1]))

                datapoint_rearranged = np.squeeze(np.swapaxes(val[:,[datapoint, datapoint+num_datapoints_per_category],:], 0, 2))
                total_entries = datapoint_rearranged.shape[-1]
                datapoint_rearranged = datapoint_rearranged.reshape((num_models, -1))
                # plt.title(f'{window} window, {point_names[datapoint]}')
                colors = ['#1f77b4', '#ff7f0e']*total_entries
                for i in range(datapoint_rearranged.shape[-1]):
                    plt.plot(x_models, datapoint_rearranged[:, i], colors[i])
        rank_corrs.append(this_window_rank_corrs)

        # plt.show()
        save_name = os.path.join(evo_dir, f'{name}')
        plt.tight_layout()
        plt.savefig(f'{save_name}.jpg')
        plt.savefig(f'{save_name}.pgf')

    rank_corrs = np.array(rank_corrs)
    name = f'{prefix}rank_error_correlations'
    plt.figure(num=name, figsize=(actual_width,height))
    rows = len(chosen_prediction_scalars)
    cols = 1

    # x_ticks = (-10, 0, 2)
    # x_ticks = (0,5,1)
    x_ticks = (0,1,.25)
    y_ticks = (0, .75, .2)

    for pred_ind, prediction_epoch in enumerate(chosen_prediction_scalars):
        plt.subplot(rows, cols, pred_ind+1)
        plt.title(f'{prediction_epoch} prediction scalar')
        plt.xticks(np.arange(x_ticks[0], x_ticks[1], x_ticks[2]))
        plt.xlim((x_ticks[0], x_ticks[1]))
        plt.yticks(np.arange(y_ticks[0], y_ticks[1], y_ticks[2]))
        plt.ylim((y_ticks[0], y_ticks[1]))

        # plt.plot(np.log2(chosen_windows), rank_corrs[pred_ind,:,0, :])
        # plt.plot(np.arange(0,len(chosen_windows)), rank_corrs[pred_ind,:,0, :])
        plt.plot(chosen_windows, rank_corrs[pred_ind,:,0, :])

    save_name = os.path.join(evo_dir, f'{name}')
    plt.tight_layout()
    plt.savefig(f'{save_name}.jpg')
    plt.savefig(f'{save_name}.pgf')


def analyze_nasnet_archs():
    def load_accuracies(dir_name):

        dir_path = os.path.join(evo_dir, dir_name)
        model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

        meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
        meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]
        meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])

        accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models])

        return accuracies

    dir_names = ['zs_small_3x3_16e_24f', 'zs_small_3x3_32e_24f', 'zs_medium_5x3_16e_24f', 'zs_medium_5x3_16e_32f', 'zs_medium_6x3_16e_32f', 'zs_standard_6x3_32e_32f']

    accuracies = [load_accuracies(x) for x in dir_names]

    _analyze_multiple(accuracies, prefix='nasnet_')

def analyze_nasbench_archs(sample_size=16):
    final_accuracies_filename = 'nas_bench_201_cifar10_test_accuracies_200.npy'
    if not os.path.exists(final_accuracies_filename):
        api = get_nasbench201_api()
        generate_nasbench201_final_properties_file(api, final_accuracies_filename)

    accuracies = [np.load(final_accuracies_filename)[x*sample_size:(x+1)*sample_size, :, 0] for x in range(4)]
    _analyze_multiple(accuracies, prefix=f'nasbench_{sample_size}_')

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
    # multi_model_test('zs_standard_6x3_32e_32f', num_models=num_models, hparams=medium_params(32), emb_queue=embeddings)
    # multi_model_test('zs_medium_6x3_16e_32f', num_models=num_models, hparams=medium_params(16), emb_queue=embeddings)
    multi_model_test('zs_small_3x3_16e_24f', num_models=num_models, hparams=small_params(16), emb_queue=embeddings)
    multi_model_test('zs_long_16', num_models=num_models, hparams=long_params(), emb_queue=long_embeddings)

def get_nasbench201_api():
    file_name = 'NAS-Bench-201-v1_1-096897.pth'
    file_path = os.path.join(res_dir, file_name)

    if os.path.exists(file_path):
        api = nasapi(file_path,verbose=False)
        return api
    else:
        return None
def generate_nasbench201_final_properties_file(api, output_filename):
    size = 200
    key = 888

    def qu(ind):
        return api.query_by_index(ind, 'cifar10', f'{size}')[key]

    def props(entry, ind):
        eval = entry.get_eval('ori-test', ind)
        acc = eval['accuracy'] / 100
        duration = eval['all_time']
        return acc, duration

    all_props = []

    for i in range(len(api)):
        query = qu(i)
        this_candidate = []
        for e in range(size):
            p = props(query, e)
            this_candidate.append(p)
        all_props.append(this_candidate)

    print(f'shape: {np.array(all_props).shape}')
    np.save(output_filename, all_props)
def analyze_nasbench201_final_properties(output_filename, show = False):
    props = np.load(output_filename)

    final_accs = props[:, -1, 0]
    final_times = props[:, -1, 1]

    print(f'final props shape: {props.shape}')

    anderson = scipy.stats.anderson(final_accs)
    print(f'anderson: {anderson}')
    final_accs_boxcox, _ = scipy.stats.boxcox(final_accs)
    anderson = scipy.stats.anderson(final_accs_boxcox)
    print(f'anderson after boxcox: {anderson}')

    day_scalar = 60 * 60 * 24

    time_mean = np.mean(final_times)
    expected_explore_time = time_mean * props.shape[0]
    expected_explore_time_days = expected_explore_time / day_scalar
    print(f'accuracy mean: {np.mean(final_accs)}')
    print(f'time mean: {time_mean}')
    print(f'expected explore time: {expected_explore_time}s == {expected_explore_time_days}d')

    if show:
        xvals = [x for x in range(final_accs.shape[0])]

        plt.subplot(1, 1, 1)
        plt.scatter(xvals, final_accs)
        plt.show()

    return expected_explore_time
def analyze_and_show_nasbench201_final_properties():
    final_accuracies_filename = 'nas_bench_201_cifar10_test_accuracies_200.npy'
    if not os.path.exists(final_accuracies_filename):
        api = get_nasbench201_api()
        generate_nasbench201_final_properties_file(api, final_accuracies_filename)
    analyze_nasbench201_final_properties(final_accuracies_filename, True)


def run_nas_api_evo(api, prediction_epoch_scalar, window_scalar, time_budget, use_zscore):
    num_api_epochs = 200
    num_api_archs = len(api)

    actual_prediction_epoch = int(num_api_epochs * prediction_epoch_scalar)
    actual_window = max(int(actual_prediction_epoch * window_scalar), 1)

    def mutate_matrix(matrix):
        row = int(np.random.random()*3)+1
        col = int(np.random.random()*row)
        op = np.random.randint(0, 5)
        new_matrix = matrix.copy()
        new_matrix[row,col] = op
        return new_matrix

    def matrix_to_arch(m):
        space = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        def st(op_ind, node_ind):
            return f'{space[int(op_ind)]}~{node_ind}'
        return f'|{st(m[1][0], 0)}|+|{st(m[2][0], 0)}|{st(m[2][1], 1)}|+|{st(m[3][0], 0)}|{st(m[3][1], 1)}|{st(m[3][2], 2)}|'

    def mutate_arch(arch):
        matrix = api.str2matrix(arch)
        new_matrix = mutate_matrix(matrix)
        return matrix_to_arch(new_matrix)

    def eval_arch(arch, available_time):
        completed_eval = True

        index = api.query_index_by_arch(arch)
        info = api.query_by_index(index, 'cifar10', hp='200')[888] # available seeds are 777, 888, and 999
        arch_accuracies = [0 for _ in range(actual_prediction_epoch)]
        for i in range(actual_prediction_epoch):
            results = info.get_eval('ori-test', i)
            if results['all_time'] > available_time:
                completed_eval = False
                break
            else:
                arch_accuracies[i] = results['accuracy'] / 100.

        utilized_time = info.get_eval('ori-test', actual_prediction_epoch - 1)['all_time']

        actual_final_accuracy = info.get_eval('ori-test')['accuracy'] / 100.
        return completed_eval, arch_accuracies, utilized_time, actual_final_accuracy

    population = []
    history = []

    population_size = 100
    frame_size = 25

    total_utilized_time = 0

    def available_time():
        return time_budget - total_utilized_time

    def eval_and_add_new_arch(arch):
        nonlocal total_utilized_time
        for candidate in history:
            if candidate[0] == arch:
                utilized_time = candidate[3]
                if utilized_time < available_time():
                    history.append(candidate)
                    population.append((candidate[:2]))
                    total_utilized_time += utilized_time
                    return True
                else:
                    return False

        completed, accuracies, utilized_time, actual_final_accuracy = eval_arch(arch, available_time())
        if completed:
            total_utilized_time += utilized_time
            history.append((arch, accuracies, actual_final_accuracy, utilized_time))
            population.append((arch, accuracies))
        return completed

    # establish initial population
    initial_population_indexes = random_exclusive(num_api_archs, population_size)
    for i in range(population_size):
        # print(f'evaluating initial population {i}/{population_size}')
        arch = api.query_meta_info_by_index(initial_population_indexes[i]).arch_str
        if not eval_and_add_new_arch(arch):
            print('ran out of time during initial population evaluation')
            return 'invalid', 0., [('INVALID',[0], 0)]

    def calculate_best_index(accuracies) -> int:
        if use_zscore:
            zscores = scipy.stats.zscore(accuracies, axis=0)
            windowed_zscores = np.array([x[-actual_window:] for x in zscores])
            mean_zscores = np.mean(windowed_zscores, axis=1)
            result = int(np.argmax(mean_zscores))
            return result
        else:
            ranks = get_ranks(accuracies)
            windowed_ranks = ranks[:, -actual_window:]
            mean_ranks = np.mean(windowed_ranks, axis=1)
            result = int(np.argmax(mean_ranks))
            return result

    # do evolution
    while available_time() > 0:
        frame = [population[i] for i in random_exclusive(population_size, frame_size)]
        accs = [x[1] for x in frame]
        best_index = calculate_best_index(accs)
        new_arch = mutate_arch(frame[best_index][0])
        if not eval_and_add_new_arch(new_arch):
            break
        del population[0]

    sorted_history = sorted(history, key=lambda x: x[1][-1])

    best_arch = sorted_history[-1][0]
    # best_arch_performance = sorted_history[-1][1]
    best_arch_final_performance = api.query_by_index(api.query_index_by_arch(best_arch), 'cifar10', '200')[888].get_eval('ori-test')['accuracy']

    # print(f'completed evaluation with {len(history)} candidates. best accuracy: {best_arch_final_performance}, arch: {best_arch}')

    return best_arch, best_arch_final_performance, history
def get_full_sim_output_filename(sim_name, prediction_scalar, window):
    sim_dir = 'sim_results'
    if not os.path.exists(os.path.join(evo_dir, sim_dir)):
        os.makedirs(os.path.join(evo_dir, sim_dir))
    return os.path.join(evo_dir, sim_dir, f'sim_results_{sim_name}_scalar{prediction_scalar}_window{window}.json')
def test_nasbench201(api, time_budget, num_sims, sim_name, use_zscore):
    windows = [1, .5, .25, .001]
    prediction_scalars = [1, .5, .25, .125]

    for prediction_scalar in prediction_scalars:
        for window in windows:
            filename = get_full_sim_output_filename(sim_name, prediction_scalar, window)
            if not os.path.exists(filename):
                np.random.seed(0)
                histories = []
                for sim in range(num_sims):
                    duration = time.time()
                    arch, perf, history = run_nas_api_evo(api, prediction_scalar, window, time_budget, use_zscore)
                    l = len(history)
                    duration = time.time() - duration
                    histories.append(history)

                    print(f'window: {window}, scalar: {prediction_scalar}, sim: {sim}/{num_sims}, perf: {perf}, len: {l}, duration: {int(duration*1000)/1000}')
                print()

                with open(filename, 'w+') as fl:
                    json.dump(histories, fl, indent=4)
def run_nasbench201_sims(api):
    final_accuracies_filename = os.path.join(res_dir, 'nas_bench_201_cifar10_test_accuracies_200.npy')
    if not os.path.exists(final_accuracies_filename):
        generate_nasbench201_final_properties_file(api, final_accuracies_filename)
    expected_explore_time = analyze_nasbench201_final_properties(final_accuracies_filename)
    allotted_explore_time = expected_explore_time / 8
    num_sims = 64

    test_nasbench201(api, allotted_explore_time, num_sims, f'{num_sims}_sims_zscore', True)

    # test_nasbench201(api, allotted_explore_time, num_sims, f'{num_sims}_sims_rank', False)
def analyze_nasbench201_sim_results():
    windows = [1, .5, .25, .001]
    prediction_scalars = [1, .5, .25, .125]

    # sim_name = '64_sims_rank'
    sim_name = '64_sims_zscore'

    data = []
    for prediction_scalar in prediction_scalars:
        for window in windows:
            filename = get_full_sim_output_filename(sim_name, prediction_scalar, window)
            print(f'loading {filename}')
            with open(filename, 'r') as fl:
                histories = json.load(fl)
                data.append((prediction_scalar, window, histories))

    num_slices = 100

    def get_actual_final_accuracy_for_predicted_best(history):
        if len(history) == 0:
            return 0.
        else:
            sorted_history = sorted(history, key=lambda x: x[1][-1])
            return sorted_history[-1][2]

    all_best_in_slices = []
    for configuration in data:
        slices_for_history = [0 for _ in range(num_slices)]
        num_histories = len(configuration[-1])
        for history in configuration[-1]:
            for sl in range(num_slices):
                slice_as_point_in_history = int(len(history) * (sl/num_slices))
                best_at_slice = get_actual_final_accuracy_for_predicted_best(history[:slice_as_point_in_history])
                slices_for_history[sl] += best_at_slice
            #find the one that it thinks is the best, and get its actual final accuracy, at each timestep in slices

        slices_for_history = [x / num_histories for x in slices_for_history[1:]]
        all_best_in_slices.append(slices_for_history)

    x_vals = [x for x in range(num_slices - 1)]

    best_configurations = []
    config_indexes = []
    for sl in range(num_slices - 1):
        print(f'processing slice {sl}')
        at_this_slice = [x[sl] for x in all_best_in_slices]
        best_ind = int(np.argmax(at_this_slice))
        best_configurations.append((data[best_ind][0], data[best_ind][1]))
        config_indexes.append(best_ind)

    plt.subplot(1,1,1)
    plt.plot(x_vals, config_indexes)
    plt.show()

    y_vals = np.swapaxes(np.array(all_best_in_slices), 0, 1)

    num_scalars = 4
    for i in range(num_scalars):
        plt.subplot(num_scalars, 1, i+1)
        plt.plot(x_vals, y_vals[:,(i*4):((i+1)*4)])

    plt.show()

def test_spearman():
    spcf = []
    avgd = []
    for i in range(10000):
        size = 1000
        vals = np.random.randint(0,size,size=size)
        keys = np.arange(0,size)
        spcf.append(spearman_coef_distinct(keys,vals))

        avgd.append(np.mean(np.abs(vals-keys))/size)
        # print(avg_distance)
    print(np.mean(spcf))
    print(np.mean(avgd))


if __name__ == '__main__':

    # analyze_nasbench201()
    # multi_config_test()

    # analyze_and_show_nasbench201_final_properties()

    # api = get_nasbench201_api()
    # run_nasbench201_sims(api)
    # analyze_nasbench201_sim_results()

    # test_spearman()
    analyze_nasbench_archs(16)
    # analyze_nasbench_archs(100)
    analyze_nasnet_archs()


    # analyze_stuff('zs_set_1\\zs_medium')
    # analyze_stuff('zs_small')
    # analyze_stuff('zs_medium')
    # analyze_slices('zs_small')
    # analyze_slices('zs_medium')
    # analyze_slices('zs_set_1\\zs_medium')

