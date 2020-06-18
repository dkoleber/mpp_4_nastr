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

def multi_model_test():
    hyperparameters = Hyperparameters()
    dataset = ImageDataset.get_cifar10()

    embeddings_queue = [
        MetaModel.get_nasnet_embedding(),
        MetaModel.get_s1_embedding(),
        MetaModel.get_identity_embedding(),
        MetaModel.get_m1_sep3_embedding(),
        MetaModel.get_m1_sep7_embedding(),
        MetaModel.get_m1_sep3_serial_embedding(),
    ]

    dir_path = os.path.join(evo_dir, 'static_analysis_samples')
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


    count = 0
    while True:
        print(f'Evaluating non-queued model {count}')
        count += 1
        eval_model()

def analyze_stuff():
    dir_path = os.path.join(evo_dir, 'static_analysis_samples')
    model_names = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

    meta_models = [MetaModel.load(dir_path, x, False) for x in model_names]

    accuracies = np.array([np.array(x.metrics.metrics['accuracy']) for x in meta_models if len(x.metrics.metrics['accuracy']) == x.hyperparameters.parameters['TRAIN_ITERATIONS']])

    num_samples = accuracies.shape[0]
    num_datapoints_per_sample = accuracies.shape[1]

    x = [x for x in range(num_datapoints_per_sample)]

    zscores = scipy.stats.zscore(accuracies, axis=0)


    avg_z_score = np.mean(zscores, axis=1)
    coorelation = np.corrcoef(avg_z_score, zscores[:, -1])
    print(f'avg z score coorelation with final z score: {coorelation[0,1]}')
    z_mean_over_time = np.array([[np.mean(np.array(x[:i])) if i > 0 else 0.001 for i in range(num_datapoints_per_sample)] for x in zscores])
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
    print(reformed_data_points)



    # reformed_data_points = np.array([sum([[v for k,v in cell.items() if type(v) is not dict] for cell in x],[]) for x in data_points])

    # reformed_data_points = reformed_data_points[:, :2]


    accuracy = np.array([x.metrics.metrics['accuracy'][-1] for x in meta_models]).astype(np.float64).reshape((-1, 1))
    accuracy, reformed_data_points = zip(*sorted(zip(accuracy, reformed_data_points), key=lambda x: x[0]))
    accuracy = np.array(accuracy)
    reformed_data_points = np.array(reformed_data_points)

    # print(reformed_data_points.shape)
    print(reformed_data_points)
    print(accuracy)

    # reduced_data_points = np.array([(x[0]['residual_ratio'], x[0]['num_outputs']) for x in data_points]).astype(np.float64).reshape((-1, 2)) #, x[1]['residual_ratio'], x[1]['num_outputs']

    #
    # print(data_points)
    # print(reduced_data_points)
    # print(reduced_data_points.shape)
    # print(accuracy.shape)

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
                    feature = features[i]
                    for power in range(p):
                        feature *= features[i]
                    features.append(feature)



            append_squared_feature(0, 2) #residual ratio, cell 0
            # append_squared_feature(1, 2)
            append_squared_feature(8, 2) #residual ratio, cell 1
            # append_squared_feature(9, 2)
            # append_squared_feature(3)

            outputs = [features[i] * self.weights[i] for i in range(len(features))]

            return tf.reduce_sum(outputs, axis=0) + self.b


    class LinRegEnsemble:
        def __init__(self):
            self.models = [LinRegModel() for x in range(8)]
            self.weights = sum([m.weights for m in self.models], [])
            self.vars = sum([m.vars for m in self.models], [])
        def __call__(self, x):
            outputs = [m(x) for m in self.models]
            return tf.reduce_mean(outputs, axis=0)

    model = LinRegEnsemble()
    optimizer = tf.keras.optimizers.SGD(.1)

    def train_step(data_points, nas_accuracy_actual):
        with tf.GradientTape() as tape:
            nas_accuracy_prediction = model(data_points)
            accuracy_loss = tf.reduce_mean(tf.square(nas_accuracy_actual - nas_accuracy_prediction))
            l2_loss = (tf.reduce_sum(tf.square(model.weights)) / 2) * .05
            loss = accuracy_loss + l2_loss
            print(f'total loss: {loss}, accuracy loss: {accuracy_loss}, l2 loss: {l2_loss}')
            # print(f'actuals: {nas_accuracy_actual}')
            # print(f'preds: {nas_accuracy_prediction}')


        variables = model.vars
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables))


    for i in range(100):
        train_step(reformed_data_points, accuracy)

    nas_accuracy_prediction = model(reformed_data_points)

    lines_to_plot = [accuracy, nas_accuracy_prediction]
    # lines_to_plot = [reformed_data_points[:,:1], reformed_data_points[:,2:3]]
    # lines_to_plot = [reformed_data_points[:,1:2], reformed_data_points[:,3:4]]


    acc = np.concatenate(lines_to_plot).reshape((len(accuracy), len(lines_to_plot)), order='F')
    x = [x for x in range(len(accuracy))]
    plt.subplot(1, 1, 1)
    plt.plot(x, acc)

    plt.title('x v y')
    plt.xlabel('x')
    plt.ylabel('y')

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



if __name__ == '__main__':
    # test_residual_ratio()
    multi_model_test()
    # analyze_stuff()
    # linreg_on_static_measurements()
    # linreg_test()