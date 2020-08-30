import os
import sys
from pathlib import Path

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import string
from pandas import DataFrame

def is_linux():
    return sys.platform == 'linux'

if is_linux():
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

from cycler import cycler

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(HERE / '../'))

from Utils import list_contains_list
from model.MetaModel import  *

from nas_201_api import NASBench201API as nasapi


def save_plt(s):
    if is_linux():
        plt.savefig(f'{s}.pgf')
    else:
        plt.savefig(f'{s}.jpg')

def remove_lowercase(s):
    table = str.maketrans('', '', string.ascii_lowercase)
    return s.translate(table)

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


def zm_rank(accuracies, window):
    zscores = scipy.stats.zscore(accuracies, axis=0)
    windowed_zscores = zscores[:, -window:]
    mean_zscores = np.mean(windowed_zscores, axis=1)
    result = int(np.argmax(mean_zscores))
    return result, get_ranks(mean_zscores)

def rm_rank(accuracies, window):
    ranks = get_ranks(accuracies)
    windowed_ranks = ranks[:, -window:]
    mean_ranks = np.mean(windowed_ranks, axis=1)
    result = int(np.argmax(mean_ranks))
    return result, get_ranks(mean_ranks)

def _analyze_multiple(accuracies, num_simulations = 32, prefix=''):

    num_experiments = len(accuracies)
    num_epochs = [len(x[0]) for x in accuracies]
    num_models = len(accuracies[0])

    chosen_windows = [1, .5, .25, .001]
    chosen_prediction_scalars = [1, .5, .25, .125]
    num_datapoints = 3
    num_eval_metrics = 2

    def gen_data():

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


            rank_corrs = [[] for _ in range(num_eval_metrics)]
            for sim_ind, simulation in enumerate(indexes):
                print(f'simulating: {sim_ind} of {num_simulations}')
                results.append([[] for _ in range(num_datapoints)])
                ranks = [[] for _ in range(num_eval_metrics)]
                rank_distances = [[] for _ in range(num_eval_metrics)]
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

                    avg_zm_rank_error_norm = average_zm_rank_distance/(num_models_so_far+1)
                    avg_rm_rank_error_norm = average_rm_rank_distance/(num_models_so_far+1)

                    this_zm_rank_error_norm = this_zm_rank_distance / (num_models_so_far + 1)
                    this_rm_rank_error_norm = this_rm_rank_distance/(num_models_so_far+1)

                    results[-1][0].append((avg_zm_rank_error_norm, avg_rm_rank_error_norm))
                    results[-1][1].append((this_zm_rank_error_norm, this_rm_rank_error_norm))
                    results[-1][2].append((zscore_spearman, rm_spearman))

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

        # COLLECT DATA
        metrics = []
        rank_corrs = []
        for pred in chosen_prediction_scalars:
            window_m = []
            window_r = []
            for window in chosen_windows:
                results = [predict_with_window(acc, window, pred) for acc in accuracies]
                m = [x[0] for x in results]
                print(f's: {np.array(m).shape}')
                window_m.append(m)
                window_r.append([x[1] for x in results])
            metrics.append(window_m)
            rank_corrs.append(window_r)

        metrics = np.array(metrics)
        rank_corrs = np.array(rank_corrs)

        return metrics, rank_corrs

    # LOAD/GENERATE DATA
    sim_data_base = os.path.join(res_dir, f'{prefix}all_sim_data')
    sim_paths = [f'{sim_data_base}_metrics.npy', f'{sim_data_base}_rank_corrs.npy']
    all_exist = all([os.path.exists(x) for x in sim_paths])
    metrics = None
    rank_corrs = None
    if not all_exist:
        print(f'Generating data for {prefix}')
        metrics, rank_corrs = gen_data()
        np.save(sim_paths[0], metrics)
        np.save(sim_paths[1], rank_corrs)
    else:
        metrics = np.load(sim_paths[0])
        rank_corrs = np.load(sim_paths[1])
    num_populations = metrics.shape[2]
    # print('finished loading')
    # print(f'metrics shape: {metrics.shape}') # dims = (prediction epochs, windows, population group, datapoint, steps, metric type)
    # print(f'rank corrs shape: {rank_corrs.shape}') # dims = (prediction epochs, windows, population group, metric type)

    actual_width = 5.5
    height = actual_width
    point_names = ['Average Rank Error', 'New Rank Error', 'Spearman Coef']
    metric_names = ['ZM', 'RM']
    y_ticks = [(0, .5, .1), (0, .5, .1), (0., 1.1, .2)]
    x_ticks = (0, num_models, int(num_models / 4))
    x_models = [x for x in range(num_models)]
    color_pairs = [['#1f77b4', '#ff7f0e'], ['#2ca02c', '#d62728'], ['#9467bd', '#8c564b'], ['#e377c2', '#7f7f7f']]

    convergences = []

    for pred_ind, pred in enumerate(chosen_prediction_scalars):
        num_plots = 0
        acceleration = int(1 / pred)
        fig_name = f'{prefix}{acceleration}x_acceleration'
        plt.figure(num=fig_name, figsize=(actual_width, height))
        rows = len(chosen_windows)
        cols = num_datapoints
        for window_ind, window in enumerate(chosen_windows):
            datapoint_convergences = []
            for datapoint in range(num_datapoints):
                num_plots += 1
                ax = plt.subplot(rows, cols, num_plots)
                if datapoint == 0:
                    ax.set_ylabel(f'{window} window')
                if window_ind == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel(point_names[datapoint])

                converge_cutoff = int(num_models/4)
                converge = metrics[pred_ind, window_ind, :, datapoint, -converge_cutoff:, :].mean(axis=0).mean(axis=0)

                datapoint_convergences.extend(converge)
                if converge[1] == 0:
                    datapoint_convergences.append(0)
                else:
                    datapoint_convergences.append(converge[0]/converge[1])

                for population_ind in range(num_populations):
                    for metric_ind in range(num_eval_metrics):
                        to_plot = metrics[pred_ind,window_ind,population_ind,datapoint,:,metric_ind]
                        c = cycler(color = [color_pairs[0][metric_ind]])
                        ax.set_prop_cycle(c)

                        ax.plot(x_models, to_plot)
                        plt.xticks(np.arange(x_ticks[0], x_ticks[1], x_ticks[2]))
                        plt.xlim((x_ticks[0], x_ticks[1]))
                        plt.yticks(np.arange(y_ticks[datapoint][0], y_ticks[datapoint][1], y_ticks[datapoint][2]))
                        plt.ylim((y_ticks[datapoint][0], y_ticks[datapoint][1]))
            convergences.append((pred, window,)+tuple(datapoint_convergences))

        plt.tight_layout()
        save_name = os.path.join(fig_dir, f'{fig_name}')
        print(f'saving {save_name}')
        save_plt(save_name)


    conv_frame = DataFrame(data=np.array(convergences))
    # col_names = ['Prediction Epoch', 'Window']
    # for datapoint_name in point_names:
    #     for metric_name in metric_names:
    #         col_names.append(f'{datapoint_name} {metric_name}')
    # col_names = [remove_lowercase(x) for x in col_names]
    col_names = ['PES', 'WS', 'ZM PARE', 'RM PARE', 'PARE Ratio', 'ZM PNRE', 'RM PNRE', 'PNRE Ratio', 'ZM Spr', 'RM Spr', 'Spr Ratio']
    conv_frame.to_csv(os.path.join(fig_dir, f'{prefix}convergences.csv'), header=col_names, index=False)

    x_ticks = (0, 1, .25)
    y_ticks = (-.2, .75, .2)

    name = f'{prefix}rank_error_correlations'
    plt.figure(num=name, figsize=(actual_width, height))
    rows = len(chosen_prediction_scalars)
    cols = 1
    num_plots = 0
    for pred_ind, pred in enumerate(chosen_prediction_scalars):
        num_plots += 1
        ax = plt.subplot(rows, cols, num_plots)
        ax.set_xlabel(f'Window')
        ax.set_ylabel(f'Correlation')
        acceleration = int(1 / pred)
        plt.title(f'{acceleration}x Acceleration')
        plt.xticks(np.arange(x_ticks[0], x_ticks[1], x_ticks[2]))
        plt.xlim((x_ticks[0], x_ticks[1]))
        plt.yticks(np.arange(y_ticks[0], y_ticks[1], y_ticks[2]))
        plt.ylim((y_ticks[0], y_ticks[1]))
        # for population_ind in range(num_populations):
        for metric_ind in range(num_eval_metrics):
            to_plot = rank_corrs[pred_ind, :, :, metric_ind].mean(axis=1)
            c = cycler(color=[color_pairs[0][metric_ind]])
            ax.set_prop_cycle(c)



            ax.plot(chosen_windows, to_plot)

    plt.tight_layout(pad=.1)
    save_name = os.path.join(fig_dir, f'{name}')
    save_plt(save_name)

    l = (' & ').join([str(np.round(x, 3)) for x in rank_corrs.mean(axis=3).mean(axis=2).mean(axis=1).tolist()])
    print(f'avg at epochs: {l}')

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
    final_accuracies_filename = res_dir / 'nas_bench_201_cifar10_test_accuracies_200.npy'
    if not os.path.exists(final_accuracies_filename):
        print(f'missing path: {final_accuracies_filename}')
        api = get_nasbench201_api()
        generate_nasbench201_final_properties_file(api, final_accuracies_filename)

    accuracies = [np.load(final_accuracies_filename)[x*sample_size:(x+1)*sample_size, :, 0] for x in range(4)]
    _analyze_multiple(accuracies, prefix=f'nasbench_{sample_size}_')

def train_nasnet_archs():

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
        print('Generating NASBench final properties')
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
            return zm_rank(accuracies, actual_window)[0]
        else:
            return rm_rank(accuracies, actual_window)[0]

    # do evolution
    while available_time() > 0:
        frame = [population[i] for i in random_exclusive(population_size, frame_size)]
        accs = [x[1] for x in frame]
        best_index = calculate_best_index(accs)
        new_arch = mutate_arch(frame[best_index][0])
        if not eval_and_add_new_arch(new_arch):
            break
        del population[0]

    # sorted_history = sorted(history, key=lambda x: x[1][-1])

    # best_arch = sorted_history[-1][0]
    # best_arch_performance = sorted_history[-1][1]
    # best_arch_final_performance = api.query_by_index(api.query_index_by_arch(best_arch), 'cifar10', '200')[888].get_eval('ori-test')['accuracy']

    # print(f'completed evaluation with {len(history)} candidates. best accuracy: {best_arch_final_performance}, arch: {best_arch}')

    return history
def get_full_sim_output_filename(sim_name, prediction_scalar, window):
    sim_dir = 'sim_results'
    if not os.path.exists(os.path.join(evo_dir, sim_dir)):
        os.makedirs(os.path.join(evo_dir, sim_dir))
    return os.path.join(evo_dir, sim_dir, f'sim_results_{sim_name}_scalar{prediction_scalar}_window{window}.json')
def test_nasbench201(api, time_budget, num_sims, sim_name, use_zscore, prediction_scalars, windows):
    for prediction_scalar in prediction_scalars:
        for window in windows:
            filename = get_full_sim_output_filename(sim_name, prediction_scalar, window)
            if not os.path.exists(filename):
                np.random.seed(0)
                histories = []
                for sim in range(num_sims):
                    duration = time.time()
                    history = run_nas_api_evo(api, prediction_scalar, window, time_budget, use_zscore)
                    l = len(history)
                    duration = time.time() - duration
                    histories.append(history)

                    print(f'window: {window}, scalar: {prediction_scalar}, sim: {sim}/{num_sims}, len: {l}, duration: {int(duration*1000)/1000}')
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

    windows = [1, .5, .25, .001]
    prediction_scalars = [1, .5, .25, .125]

    # test_nasbench201(api, allotted_explore_time, num_sims, f'{num_sims}_sims_zscore', True, prediction_scalars, windows)

    # test_nasbench201(api, allotted_explore_time, num_sims, f'{num_sims}_sims_rank', False, prediction_scalars, windows)

    prediction_scalars = prediction_scalars[:1]
    test_nasbench201(api, expected_explore_time, num_sims, f'{num_sims}_extended_sims_zscore', True, prediction_scalars, windows)
    test_nasbench201(api, expected_explore_time, num_sims, f'{num_sims}_extended_sims_rank', False, prediction_scalars, windows)


def analyze_nasbench201_sim_results(prediction_scalars, windows, prefix):
    num_metrics = 2
    num_sims = 64
    num_measurements = 3
    num_slices = 100
    num_epochs = 200
    running_pop_size = 100
    max_pop_size = 20000

    metric_names = [f'{prefix}sims_zscore', f'{prefix}sims_rank']

    population_sizes = None
    all_measurements = None
    actual_performances = None

    sim_dir = evo_dir / 'sim_results'
    measurements_path = sim_dir / f'{prefix}measurements.npy'
    population_sizes_path = sim_dir / f'{prefix}population_sizes.npy'
    performances_path = sim_dir / f'{prefix}actual_performances.npy'

    def gen_properties():
        population_size = np.zeros((len(prediction_scalars), len(windows), num_metrics, num_sims))
        measurements = np.zeros((len(prediction_scalars), len(windows), num_metrics, num_sims, max_pop_size, num_measurements))  # pred, window, metric, sim, measurement, slice_ind
        actual_performances_of_predicted_best = np.zeros((len(prediction_scalars), len(windows), num_metrics, num_sims, max_pop_size))

        def get_actual_performance_of_predicted_best(history):
            if len(history) == 0:
                return 0.
            else:
                sorted_history = sorted(history, key=lambda x: x[1][-1])
                return sorted_history[-1][2]

        def get_measurements_in_population(pop, metric_ind, window):
            rank_func = None
            if metric_ind == 0:
                rank_func = zm_rank
            else:
                rank_func = rm_rank

            accuracies = np.array([x[1] for x in pop])
            predicted_ranks = rank_func(accuracies, window)[1]
            actual_ranks = get_ranks(np.array([x[2] for x in pop]))

            pare = np.mean(np.abs(predicted_ranks - actual_ranks)) / predicted_ranks.shape[0]
            pnre = np.abs(predicted_ranks[-1] - actual_ranks[-1]) / predicted_ranks.shape[0]
            spr = spearman_coef_distinct(predicted_ranks, actual_ranks)

            return (pare, pnre, spr)

        for pred_ind, pred in enumerate(prediction_scalars):
            for window_ind, window in enumerate(windows):
                for metric_ind, metric_name in enumerate(metric_names):
                    filename = get_full_sim_output_filename(metric_name, pred, window)
                    print(f'loading {filename}')
                    with open(filename, 'r') as fl:
                        mixed_data = json.load(fl)
                        for sim_ind, sim_history in enumerate(mixed_data):  # history is [(arch_string, accuracies, actual_final_accuracy, utilized_time]
                            this_sim_size = len(sim_history)
                            population_size[pred_ind, window_ind, metric_ind, sim_ind] = this_sim_size
                            for i in range(len(sim_history)):
                                start_ind = max(0, i - running_pop_size + 1)
                                end_ind = i + 1
                                measurements[pred_ind, window_ind, metric_ind, sim_ind, i] = get_measurements_in_population(sim_history[start_ind:end_ind], metric_ind, max(int(window * num_epochs * pred),1))
                                actual_performances_of_predicted_best[pred_ind, window_ind, metric_ind, sim_ind, i] = get_actual_performance_of_predicted_best(sim_history[:i+1])

        np.save(measurements_path, measurements)
        np.save(population_sizes_path, population_size)
        np.save(performances_path, actual_performances_of_predicted_best)

        return measurements, population_size, actual_performances_of_predicted_best

    if not os.path.exists(population_sizes_path):
        all_measurements, population_sizes, actual_performances = gen_properties()
    else:
        all_measurements = np.load(measurements_path)       # (4, 4, 2, 64, 20000, 3)
        population_sizes = np.load(population_sizes_path)   # (4, 4, 2, 64)
        actual_performances = np.load(performances_path)    # (4, 4, 2, 64, 20000)

    actual_width = 5.5
    height = actual_width
    point_names = ['PARE', 'PNRE', 'Spearman Coef']
    metric_names = ['ZM', 'RM']
    y_ticks = [(0, .5, .1), (0, .5, .1), (0., 1.1, .2)]
    x_ticks = (0, num_slices, int(num_slices / 4))
    x_coords = np.array([x for x in range(max_pop_size)])
    color_pairs = [['#1f77b4', '#ff7f0e'], ['#2ca02c', '#d62728'], ['#9467bd', '#8c564b'], ['#e377c2', '#7f7f7f']]

    fig_name = f'{prefix}evosim_all_performances_over_size'
    plt.figure(num=fig_name, figsize=(actual_width, height))
    ax = plt.subplot(1, 1, 1, label='Performances of Predicted Best at History Sizes')
    plt.ylim(.93, .95)

    def quantize(x_vals, y_vals, size=1000, mean=True):
        actual_size = x_vals.shape[0]
        slice_inds = [int((x+1)*actual_size/size) - 1 for x in range(size)]
        new_y_vals = None
        if mean:
            new_y_vals = np.zeros((size, ) + y_vals.shape[1:])
            for i in range(size):
                this_slice = slice_inds[i]
                prev_slice = slice_inds[i-1] if i > 0 else 0
                new_y_vals[i] = y_vals[prev_slice:this_slice].mean(axis=0)
        else:
            new_y_vals = y_vals[slice_inds]
        return np.array(slice_inds), new_y_vals

    for pred_ind, pred in enumerate(prediction_scalars):
        colors = []
        colors.extend(color_pairs[0])
        colors.extend(color_pairs[1])
        c = cycler(color=colors)
        ax.set_prop_cycle(c)
        for window_ind, window in enumerate(windows):
            # actual_size = int(population_sizes[pred_ind, window_ind].mean(axis=0).mean(axis=0))
            actual_size = int(np.min(population_sizes[pred_ind, window_ind]))
            x_actual = [x for x in range(actual_size)]
            to_plot = actual_performances[pred_ind, window_ind].mean(axis=0).mean(axis=0)[:actual_size]
            x_vals, y_vals = quantize(np.array(x_actual), to_plot)
            if pred_ind == 0:
                ax.plot(x_vals, y_vals, label=f'{window} window')
            else:
                ax.plot(x_vals, y_vals)
    ax.legend(loc='lower right')
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Final Accuracy of Predicted Best Candidate in Population')

    # plt.show()
    plt.tight_layout()
    save_name = os.path.join(fig_dir, f'{fig_name}')
    save_plt(save_name)

    best_perf_at_slices = np.zeros((len(prediction_scalars) * len(windows), num_metrics, num_slices))

    tick_nums = [x for x in range(len(windows) * len(prediction_scalars))]
    tick_names = []

    for pred_ind, pred in enumerate(prediction_scalars):
        for window_ind, window in enumerate(windows):
            tick_names.append(f'{pred} PES, {window} WS')
            for metric_ind in range(num_metrics):
                best_per_sim = np.zeros((num_sims, num_slices))
                for sim_ind in range(num_sims):
                    size =  int(population_sizes[pred_ind, window_ind, metric_ind, sim_ind])
                    slice_inds = [int((x+1)*size/num_slices) - 1 for x in range(num_slices)]
                    for index, sl in enumerate(slice_inds):
                        best_per_sim[sim_ind,index] = actual_performances[pred_ind, window_ind, metric_ind, sim_ind, sl]
                best_perf_at_slices[pred_ind * len(prediction_scalars) + window_ind, metric_ind] = best_per_sim.mean(axis=0)

    best_slices = best_perf_at_slices.argmax(axis=0)
    best_slices = np.swapaxes(best_slices, 0, 1)
    x_slices = [x for x in range(num_slices)]
    fig_name = f'{prefix}evosim_all_performances_over_slices'
    plt.figure(num=fig_name, figsize=(actual_width, height))
    ax = plt.subplot(1, 1, 1, label='Best Combination at Time Slices')
    plt.yticks(tick_nums, tick_names)
    plt.ylim(tick_nums[0], tick_nums[-1])
    ax.plot(x_slices, best_slices)
    ax.set_xlabel('Time Utilized as Percentage of Total Budget')
    ax.set_ylabel('Configuration with Best Final Accuracy of Predicted Best Candidate')
    plt.tight_layout()
    save_name = os.path.join(fig_dir, f'{fig_name}')
    save_plt(save_name)

    # ticks = [0, 1, 2]
    # labels = ["a", "b", "c"]
    # plt.figure()
    # plt.xticks(ticks, labels)
    # plt.show()

    datapoint_y_labels = ['Error', 'Error', 'Spr Coef']
    datapoint_x_label = 'Population Size'

    convergences = []

    for pred_ind, pred in enumerate(prediction_scalars):
        num_plots = 0
        acceleration = int(1 / pred)
        fig_name = f'{prefix}evosim_{acceleration}x_acceleration'
        plt.figure(num=fig_name, figsize=(actual_width, height))
        rows = len(windows)
        cols = num_measurements
        for window_ind, window in enumerate(windows):
            datapoint_convergences = []
            for datapoint in range(num_measurements):
                num_plots += 1
                ax = plt.subplot(rows, cols, num_plots)
                if datapoint == 0:
                    label_name = f'{window} window\n'
                    if window_ind == len(windows) - 1:
                        label_name += 'Error'
                    ax.set_ylabel(label_name)
                if window_ind == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel(point_names[datapoint])
                if window_ind == len(windows) - 1:
                    if datapoint != 0:
                        ax.set_ylabel(datapoint_y_labels[datapoint])
                    ax.set_xlabel(datapoint_x_label)


                this_datapoint_convergences = []
                for metric_ind in range(num_metrics):
                    min_pop_size = int(np.min(population_sizes[pred_ind, window_ind, metric_ind, :]))
                    to_plot = all_measurements[pred_ind, window_ind, metric_ind, :, :min_pop_size, datapoint].mean(axis=0)
                    actual_x_coords = x_coords[:min_pop_size]
                    c = cycler(color=[color_pairs[0][metric_ind]])
                    ax.set_prop_cycle(c)
                    x_vals, y_vals = quantize(np.array(actual_x_coords), to_plot)
                    ax.plot(x_vals, y_vals, label=metric_names[metric_ind] )
                    plt.yticks(np.arange(y_ticks[datapoint][0], y_ticks[datapoint][1], y_ticks[datapoint][2]))
                    plt.ylim((y_ticks[datapoint][0], y_ticks[datapoint][1]))


                    converge_cutoff = int(min_pop_size / 4)
                    converge = to_plot[-converge_cutoff:].mean(axis=0)
                    this_datapoint_convergences.append(converge)
                if this_datapoint_convergences[1] != 0:
                    this_datapoint_convergences.append((this_datapoint_convergences[0]/this_datapoint_convergences[1]))
                else:
                    this_datapoint_convergences.append(0)
                datapoint_convergences.extend(this_datapoint_convergences)
                if datapoint == 0 and window_ind == 0:
                    ax.legend(loc='upper left', prop={'size': 6})


                    # for population_ind in range(num_sims):
                    #     actual_pop_size = int(population_sizes[pred_ind, window_ind, metric_ind, population_ind])
                    #     # print(f'actual pop {actual_pop_size}')
                    #     actual_x_coords = x_coords[:actual_pop_size]
                    #     to_plot = all_measurements[pred_ind, window_ind, metric_ind, population_ind, :actual_pop_size, datapoint]
                    #     c = cycler(color=[color_pairs[0][metric_ind]])
                    #     ax.set_prop_cycle(c)
                    #
                    #     ax.plot(actual_x_coords, to_plot)
                    # print(f'{datapoint} {to_plot[:16]}')
                    # plt.xticks(np.arange(x_ticks[0], x_ticks[1], x_ticks[2]))
                    # plt.xlim((x_ticks[0], x_ticks[1]))
            convergences.append((pred, window,) + tuple(datapoint_convergences))


        plt.tight_layout()
        save_name = os.path.join(fig_dir, f'{fig_name}')
        save_plt(save_name)

    conv_frame = DataFrame(data=np.array(convergences))
    col_names = ['PES', 'WS', 'ZM PARE', 'RM PARE', 'PARE Ratio', 'ZM PNRE', 'RM PNRE', 'PNRE Ratio', 'ZM Spr', 'RM Spr', 'Spr Ratio']
    conv_frame.to_csv(os.path.join(fig_dir, f'{prefix}full_sim_convergences.csv'), header=col_names, index=False)

def analyze_all_nasbench201_sim_results():
    windows = [1, .5, .25, .001]
    prediction_scalars = [1, .5, .25, .125]
    analyze_nasbench201_sim_results(prediction_scalars, windows, '64_')

    prediction_scalars = prediction_scalars[:1]
    analyze_nasbench201_sim_results(prediction_scalars, windows, '64_extended_')


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

def get_average_nasnet_arch_size():
    # tf.compat.v1.disable_eager_execution()
    dir_names = ['zs_standard_6x3_32e_32f']#'zs_small_3x3_16e_24f', 'zs_small_3x3_32e_24f', 'zs_medium_5x3_16e_24f', 'zs_medium_5x3_16e_32f', 'zs_medium_6x3_16e_32f',

    dataset = ImageDataset.get_cifar10()

    for dir_name in dir_names:
        model_sizes = []
        dir_path = os.path.join(evo_dir, dir_name)
        model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
        for model_name in model_names:
            model = MetaModel.load(dir_path, model_name, True)
            count = model.keras_model.count_params()
            print(count)
            model_sizes.append(count)
            model.clear_model()
            tf.keras.backend.clear_session()


        print(model_sizes)
        print(f'average size for {dir_name}: {np.mean(model_sizes)}')
def get_average_nasbench_arch_size(api):
    print(f'api size: {len(api)}')
    flops = []
    for i in range(len(api)):
        flops.append(api.query_meta_info_by_index().get_compute_costs('cifar10')['params'])

    print(f'mean flops: {np.mean(flops)}')


def gen_all_graphs():
    analyze_all_nasbench201_sim_results()
    plt.close()
    analyze_nasbench_archs(16)
    plt.close()
    analyze_nasbench_archs(100)
    plt.close()
    analyze_nasnet_archs()
    plt.close()

if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1] == 'all':
        gen_all_graphs()
    else:
        # analyze_nasbench201()
        # multi_config_return()

        # analyze_and_show_nasbench201_final_properties()

        # api = get_nasbench201_api()
        # run_nasbench201_sims(api)
        # analyze_all_nasbench201_sim_results()

        # get_average_nasnet_arch_size()
        # get_average_nasbench_arch_size(api)

        # analyze_nasbench_archs(16)
        # analyze_nasbench_archs(100)
        # analyze_nasnet_archs()

        gen_all_graphs()

        # analyze_stuff('zs_set_1\\zs_medium')
        # analyze_stuff('zs_small')
        # analyze_stuff('zs_medium')
        # analyze_slices('zs_small')
        # analyze_slices('zs_medium')
        # analyze_slices('zs_set_1\\zs_medium')

