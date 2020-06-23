import os
import sys

import scipy
import scipy.stats
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from Utils import list_contains_list
from model.MetaModel import  *




def test_residual_ratio():
    hyperparameters = Hyperparameters()


    model = MetaModel(hyperparameters)
    model.populate_from_embedding(MetaModel.get_nasnet_embedding())
    model.cells[0].process_stuff()
    model.cells[1].process_stuff()

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

    for count in range(num_remaining)
        print(f'Evaluating non-queued model {count} of {num_remaining}')
        eval_model()

def analyze_stuff():
    dir_path = os.path.join(evo_dir, 'static_analysis_samples')
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])
    meta_models = meta_models[int(len(meta_models)/2):]

    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']])

    num_samples = accuracies.shape[0]
    num_datapoints_per_sample = accuracies.shape[1]

    x = [x for x in range(num_datapoints_per_sample)]

    zscores = scipy.stats.zscore(accuracies, axis=0)


    avg_z_score = np.mean(zscores, axis=1)
    coorelation = np.corrcoef(avg_z_score, zscores[:, -1])
    print(f'avg z score coorelation with final z score: {coorelation[0,1]}')
    z_mean_over_time = np.array([[np.mean(np.array(x[:i])) if i > 0 else x[i] for i in range(num_datapoints_per_sample)] for x in zscores])
    correlations_over_time_mean = np.array([np.corrcoef(z_mean_over_time[:, i], zscores[:, -1])[0,1] for i in range(num_datapoints_per_sample)])
    correlations_over_time = np.array([np.corrcoef(zscores[:, i], zscores[:, -1])[0,1] for i in range(num_datapoints_per_sample)])
    zscores = np.swapaxes(zscores, 0, 1)
    accuracies = np.swapaxes(accuracies, 0, 1)
    z_mean_over_time = np.swapaxes(z_mean_over_time, 0, 1)

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
    plt.plot(x, correlations_over_time)
    plt.title('x v zscore correlation with final z score')
    plt.xlabel('x')
    plt.ylabel('correlation')

    plt.show()

    ranked_models = [(x.model_name, x.metrics.metrics['accuracy'][-1]) for x in meta_models]
    ranked_models.sort(key=lambda x: x[1])
    print(ranked_models)


def linreg_on_static_measurements():
    dir_path = os.path.join(evo_dir, 'static_analysis_samples')
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]
    meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]

    data_points = [x.process_stuff() for x in meta_models]

    reformed_data_points = []

    for point in data_points:
        data = []
        for cell in point:
            for _, item in cell.items():
                if type(item) is dict:
                    for key, val in item.items():
                        data.append(val)
                else:
                    data.append(item)
        reformed_data_points.append(data)

    reformed_data_points = np.array(reformed_data_points)

    accuracy = np.array([x.metrics.metrics['accuracy'][-1] for x in meta_models]).astype(np.float64).reshape((-1, 1))
    accuracy, reformed_data_points = zip(*sorted(zip(accuracy, reformed_data_points), key=lambda x: x[0]))

    accuracy = scipy.stats.zscore(np.array(accuracy), axis=0)

    # print(accuracy)
    # print(accuracy.min(axis=0))
    # print(accuracy.max(axis=0))

    accuracy = (accuracy - accuracy.min(axis=0)) / (accuracy.max(axis=0) - accuracy.min(axis=0))
    # print(accuracy)
    reformed_data_points = np.array(reformed_data_points)

    for m in meta_models:
        acc = m.metrics.metrics['accuracy'][-1]
        print(f'{m.model_name} {acc}')


    ensemble_size = 4
    l2_weight = 0.001
    l1_weight = 0.001

    class LinRegModel:
        def __init__(self):
            self.weights = [tf.Variable((2*np.random.random()) - 1., dtype=tf.float64) for x in range(32)]

            self.b = tf.Variable(0, dtype=tf.float64)

            self.vars = self.weights + [self.b]

        def __call__(self, x):

            # print(f'weights: {[w.numpy() for w in self.weights]}')
            num_input_features = tf.shape(x)[1]

            features = [x[:, i:i + 1] for i in range(num_input_features)]  # maybe add or modify features?



            def append_squared_feature(i, polynomial=2):
                for p in range(1, polynomial):
                    feature = features[i].copy()**(p+1)

                    features.append(feature)





            append_squared_feature(0, 2) #residual ratio, cell 0
            append_squared_feature(1, 2)

            append_squared_feature(4, 2)  # residual ratio, cell 0
            append_squared_feature(5, 2)

            # append_squared_feature(8, 2) #residual ratio, cell 1
            # append_squared_feature(9, 2)
            # append_squared_feature(3)

            outputs = [features[i] * self.weights[i] for i in range(len(features))]

            result = tf.reduce_sum(outputs, axis=0) + self.b



            return result

    class LinRegEnsemble:
        def __init__(self):
            self.models = [LinRegModel() for x in range(ensemble_size)]
            self.weights = sum([m.weights for m in self.models], [])
            self.vars = sum([m.vars for m in self.models], [])
        def __call__(self, x):
            outputs = [m(x) for m in self.models]

            result = tf.reduce_mean(outputs, axis=0)
            return result

    class MLP:
        def __init__(self):
            reg = tf.keras.regularizers.l2(l2_weight)
            m_input = tf.keras.Input([16])

            l = tf.keras.layers.Dense(32, activation=tf.nn.relu, dtype=tf.float64, kernel_regularizer=reg)
            self.losses = l.losses
            layer = l(m_input)
            l = tf.keras.layers.Dense(32, activation=tf.nn.relu, dtype=tf.float64, kernel_regularizer=reg)
            self.losses += l.losses
            layer =  l(layer)
            l = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, dtype=tf.float64, kernel_regularizer=reg)
            self.losses += l.losses
            layer = l(layer)
            self.model = tf.keras.models.Model(inputs=m_input, outputs=layer)
            self.vars = self.model.trainable_variables
            self.weights = self.vars
        def __call__(self, x):
            return self.model(x)

    # model = MLP()
    # model = LinRegModel()
    model = LinRegEnsemble()
    optimizer = tf.keras.optimizers.SGD(.1)

    def write_same_line(s):
        blanks = ' '*100
        sys.stdout.write(f'\r{blanks}')
        sys.stdout.flush()
        sys.stdout.write(f'\r')
        sys.stdout.flush()
        sys.stdout.write(s)
        sys.stdout.flush()

    def train_step(data_points, nas_accuracy_actual):
        with tf.GradientTape() as tape:
            nas_accuracy_prediction = model(data_points)
            accuracy_loss = tf.reduce_mean(tf.square(nas_accuracy_actual - nas_accuracy_prediction))
            l2_loss = (tf.reduce_sum(tf.square(model.weights)) / 2) * l2_weight
            # l2_loss = model.losses
            # l2_loss = 0
            l1_loss = tf.reduce_sum(tf.abs(model.weights)) * l1_weight
            # reg_loss = l2_loss
            reg_loss = l1_loss
            loss = accuracy_loss + reg_loss
            write_same_line(f'total loss: {loss}, accuracy loss: {accuracy_loss}, reg loss: {reg_loss}')
            # print(f'actuals: {nas_accuracy_actual}')
            # print(f'preds: {nas_accuracy_prediction}')


        variables = model.vars
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables))


    rf_dp = reformed_data_points.copy()
    mask = np.ones(rf_dp.shape, dtype=bool)
    mask[:, [2,3,4,5,10,11,12,13]] = False
    rf_dp = np.reshape(rf_dp[mask], (reformed_data_points.shape[0], -1))

    for i in range(300):
        train_step(rf_dp.copy(), accuracy)
    print()



    nas_accuracy_prediction = model(rf_dp.copy())

    x = np.array([x for x in range(len(accuracy))])
    # lines_to_plot = [accuracy, nas_accuracy_prediction]
    # acc = np.concatenate(lines_to_plot).reshape((len(accuracy), len(lines_to_plot)), order='F')

    accuracy_v_residual_ratio = list(zip(accuracy, reformed_data_points[:,0], reformed_data_points[:,8]))
    accuracy_v_num_outputs = list(zip(accuracy, reformed_data_points[:,1], reformed_data_points[:,9]))
    accuracy_v_shortest_path_spread = list(zip(accuracy, reformed_data_points[:, 2], reformed_data_points[:, 10]))
    accuracy_v_shortest_path_power = list(zip(accuracy, reformed_data_points[:, 3], reformed_data_points[:, 11]))
    accuracy_v_avg_path_spread = list(zip(accuracy, reformed_data_points[:, 4], reformed_data_points[:, 12]))
    accuracy_v_avg_path_power = list(zip(accuracy, reformed_data_points[:, 5], reformed_data_points[:, 13]))
    accuracy_v_longest_path_spread = list(zip(accuracy, reformed_data_points[:, 6], reformed_data_points[:, 14]))
    accuracy_v_longest_path_power = list(zip(accuracy, reformed_data_points[:, 7], reformed_data_points[:, 15]))

    accuracy_vs_predicted = list(zip(accuracy, nas_accuracy_prediction[:,0]))

    # print(reformed_data_points)

    cols = 3
    rows = 3


    plt.subplot(rows, cols, 1)
    plt.plot(x, accuracy_v_residual_ratio)
    plt.title('accuracy v residual ratio')
    # plt.xlabel('x')
    # plt.ylabel('y')

    plt.subplot(rows, cols, 2)
    plt.plot(x, accuracy_v_num_outputs)
    plt.title('accuracy v num outputs')

    plt.subplot(rows, cols, 3)
    plt.plot(x, accuracy_vs_predicted)
    plt.title('accuracy v prediction')

    plt.subplot(rows, cols, 4)
    plt.plot(x, accuracy_v_shortest_path_spread)
    plt.title('accuracy v shortest path spread')

    plt.subplot(rows, cols, 5)
    plt.plot(x, accuracy_v_avg_path_spread)
    plt.title('accuracy v avg path spread')

    plt.subplot(rows, cols, 6)
    plt.plot(x, accuracy_v_longest_path_spread)
    plt.title('accuracy v longest spread')

    plt.subplot(rows, cols, 7)
    plt.plot(x, accuracy_v_shortest_path_power)
    plt.title('accuracy v shortest path power')

    plt.subplot(rows, cols, 8)
    plt.plot(x, accuracy_v_avg_path_power)
    plt.title('accuracy v avg path power')

    plt.subplot(rows, cols, 9)
    plt.plot(x, accuracy_v_longest_path_power)
    plt.title('accuracy v longest power')

    corr = np.corrcoef(accuracy[:, 0], nas_accuracy_prediction[:, 0])[0][1]
    print(f'r: {corr}, r2: {corr ** 2}')
    plt.show()


def linreg_test():
    x = np.linspace(0, 99, 100)

    accuracies = np.linspace(0., 1., 100).astype(np.float64).reshape((-1, 1))
    input_1 = np.linspace(0, 1, 100) + (np.random.randn(*(100,)) * .2)
    input_2 = np.linspace(.5, .9 ,100) + (np.random.randn(*(100,)) * .3)
    inputs = np.concatenate((input_1, input_2)).astype(np.float64).reshape((-1, 2), order='F')
    # print(accuracies.shape)
    # print(inputs.shape)
    # print(inputs)
    # print(inputs[:,:1])
    #


    class LinRegModel:
        def __init__(self):

            self.weights = [tf.Variable(np.random.normal(), dtype=tf.float64) for x in range(4)]
            self.b = tf.Variable(0, dtype=tf.float64)

            self.vars = self.weights + [self.b]

        def __call__(self, x):
            num_input_features = tf.shape(x)[1]
            print(f'shape: {tf.shape(x)}')

            features = [x[:,i:i+1] for i in range(num_input_features)] #maybe add or modify features?
            outputs = [features[i]*self.weights[i] for i in range(len(features))]

            return tf.reduce_sum(outputs) + self.b

    model = LinRegModel()
    optimizer = tf.keras.optimizers.SGD(.1)

    def train_step(data_points, nas_accuracy_actual):
        with tf.GradientTape() as tape:
            nas_accuracy_prediction = model([data_points[:, :1], data_points[:, 1:]])
            loss = tf.reduce_mean(tf.square(nas_accuracy_actual - nas_accuracy_prediction))
            print(f'loss: {loss}')
            # print(f'actuals: {nas_accuracy_actual}')
            # print(f'preds: {nas_accuracy_prediction}')


        variables = model.vars
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables))

    for i in range(100):
        train_step(inputs, accuracies)

    nas_accuracy_prediction = model(inputs)

    acc = np.concatenate([accuracies, nas_accuracy_prediction]).reshape((100, 2), order='F')

    # acc = accuracies
    # print(acc)

    plt.subplot(1, 1, 1)
    plt.plot(x, acc)

    plt.title('x v y')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def test_mutations_on_median():
    src_dir_path = os.path.join(evo_dir, 'static_analysis_samples')
    model_names = [x for x in os.listdir(src_dir_path) if os.path.isdir(os.path.join(src_dir_path, x))]
    meta_models = [MetaModel.load(src_dir_path, x, False) for x in model_names]
    meta_models = [x for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']]
    meta_models.sort(key=lambda x: x.metrics.metrics['accuracy'][-1])
    target_model = meta_models[int(len(meta_models)/2)]

    dest_dir_path = os.path.join(evo_dir, 'static_analysis_mutations')
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)

    dataset = ImageDataset.get_cifar10()

    for _ in range(16):
        model =  MetaModel.load(src_dir_path, target_model.model_name, True)
        model.hyperparameters.parameters['TRAIN_ITERATIONS'] += 1
        model.hyperparameters.parameters['TRAIN_EPOCHS'] = 1
        model.mutate()
        model.evaluate(dataset)
        model.save_metadata(dest_dir_path)
        model.save_model(dest_dir_path)
        model.generate_graph(dest_dir_path)
        tf.keras.backend.clear_session()

def multi_config_test():
    embedding_queue = [
        MetaModel.get_nasnet_embedding(),
        MetaModel.get_s1_embedding(),
        MetaModel.get_identity_embedding(),
        MetaModel.get_m1_sep3_embedding(),
        MetaModel.get_m1_sep7_embedding(),
        MetaModel.get_m1_sep3_serial_embedding(),
    ]

    epochs = 24

    def default_params() -> Hyperparameters:
        params = Hyperparameters()
        params.parameters['REDUCTION_EXPANSION_FACTOR'] = 2
        params.parameters['SGDR_EPOCHS_PER_RESTART'] = epochs
        params.parameters['TRAIN_ITERATIONS'] = epochs
        params.parameters['MAXIMUM_LEARNING_RATE'] = 0.025
        params.parameters['MINIMUM_LEARNING_RATE'] = 0
        params.parameters['DROP_PATH_TOTAL_STEPS_MULTI'] = 1
        params.parameters['BATCH_SIZE'] = 16
        return params

    def medium_params() -> Hyperparameters:
        params = default_params()
        params.parameters['TARGET_FILTER_DIMS'] = 32
        params.parameters['NORMAL_CELL_N'] = 5
        params.parameters['CELL_LAYERS'] = 3
        return params

    def small_params() -> Hyperparameters:
        params = default_params()
        params.parameters['TARGET_FILTER_DIMS'] = 24
        params.parameters['NORMAL_CELL_N'] = 3
        params.parameters['CELL_LAYERS'] = 3
        return params

    def tiny_params() -> Hyperparameters:
        params = default_params()
        params.parameters['TARGET_FILTER_DIMS'] = 16
        params.parameters['NORMAL_CELL_N'] = 2
        params.parameters['CELL_LAYERS'] = 3
        return params


    multi_model_test('zs_medium', num_models=32, hparams=medium_params())
    multi_model_test('zs_small', num_models=32, hparams=small_params())
    multi_model_test('zs_tiny', num_models=32, hparams=tiny_params())




if __name__ == '__main__':
    # test_residual_ratio()
    # multi_model_test()
    analyze_stuff()
    # linreg_on_static_measurements()
    # linreg_test()
    # test_mutations_on_median()