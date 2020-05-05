import tensorflow as tf
import numpy as np
import os
import time
from Dataset import *

from FileManagement import *
from Modelv3 import MetaModel

def make_surrogate_model():
    cell_embedding_input = tf.keras.layers.Input()
    accuracy_input = tf.keras.layers.Input()

    branch_1 = tf.keras.layers.Dense(128)(cell_embedding_input)
    branch_1 = tf.keras.layers.ReLU()(branch_1)

    # branch_2 = tf.keras.layers.Embedding()(accuracy_input)
    branch_2 = accuracy_input
    branch_2 = tf.keras.layers.LSTM(128)(branch_2)
    branch_2 = tf.keras.layers.LSTM(128)(branch_2)
    branch_2 = tf.keras.layers.Dense(128)(branch_2)
    branch_2 = tf.keras.layers.ReLU()(branch_2)

    together = tf.keras.layers.Concatenate()([branch_1, branch_2])
    together = tf.keras.layers.Dense(128)(together)
    together = tf.nn.sigmoid(together)

    model = tf.keras.Model(inputs=[cell_embedding_input, accuracy_input], outputs=together)

    return model

def train_surrogate_model():
    model = make_surrogate_model()

    data = SurrogateDataset()







def generate_sequence_data():
    dir_path = os.path.join(evo_dir, 'test_accuracy_epochs_h5')

    samples = [x for x in os.listdir(dir_path) if 'small' not in x and '.csv' not in x]

    candidates = [MetaModel.load(dir_path, x, False) for x in samples]

    embeddings = []
    accuracies = []

    for candidate in candidates:
        embeddings.append(candidate.get_embedding())
        accuracies.append(candidate.metrics.metrics['accuracy'])


    timestamp = str(time.time())
    out_dir_path = os.path.join(res_dir, f'surrogate_training_set_{timestamp}')
    os.makedirs(out_dir_path)

    np.savetxt(os.path.join(out_dir_path, 'embeddings.csv'), embeddings, delimiter=',')
    np.savetxt(os.path.join(out_dir_path, 'accuracies.csv'), accuracies, delimiter=',')


def analyze_data():
    dir_path = os.path.join(evo_dir, 'test_accuracy_epochs_h5')

    samples = [x for x in os.listdir(dir_path) if 'small' not in x]
    candidates = [MetaModel.load(dir_path, x, False) for x in samples]

    accuracies = [x.metrics.metrics['accuracy'][-1] for x in candidates]

    mean = np.average(accuracies)
    stdev = np.std(accuracies)
    max_val = max(accuracies)
    min_val = min(accuracies)

    print(f'mean: {mean}, stdev: {stdev}, max: {max_val}, min: {min_val}')



if __name__ == '__main__':
    generate_sequence_data()
    # analyze_data()