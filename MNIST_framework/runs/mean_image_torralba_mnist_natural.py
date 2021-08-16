# in this script we plot the average activation of neurons, in particular
# how much these change based on the initial value
import os
from numpy.linalg import norm
from os.path import join
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalMaxPool2D

# do we want to import some scikit-learn routine to evaluate
# the among points of the same class or from different classes?

INTERESTING_MNIST_DICT = {3: 256, 5: 128, 6: 64, 7: 28}

def search_best_id(root_path, index_list):
    """ This function, given the parameters,
    looks for the best realization
    of the experiment, among different learning rates.
    :param root_path: str of the folder containing results
    :param index_list: list of indices for the experiments with same parameters
    """
    loss_val = np.array([pd.read_csv(join(root_path,
                                          'train_%i/history.csv' % index))['val_loss'].values[-1]
                         for index in index_list])  # validation loss at last iteration
    loss_val[np.isnan(loss_val)] = np.inf
    best_id = index_list[np.argmin(loss_val)]       # best index
    return best_id


def flatten_train_json(df):
    """ We assume here that the train.json file has always the same keys.
    :param df: pandas dataframe
    :return: flatten df, each row corresponds to an experiment.
    """
    df = df.T
    dataset_keys = ['dataset_name', 'dataset_path', 'scenario',
                    'dataset_dimensions', 'n_training']
    hyper_keys = ['learning_rate', 'architecture', 'epochs',
                  'batch_size', 'optimizer',
                  'lr_at_plateau', 'reduction_factor',
                  'validation_check']
    upper_keys = ['id', 'output_path', 'train_completed']
    columns_name = dataset_keys + hyper_keys + upper_keys

    list_all_samples = []

    for i_ in range(df.shape[0]):
        list_per_sample = []
        for d_k_ in dataset_keys:
            list_per_sample.append(df['dataset'][i_][d_k_])
        for h_k_ in hyper_keys:
            list_per_sample.append(df['hyper'][i_][h_k_])
        for u_p_ in upper_keys:
            list_per_sample.append(df[u_p_][i_])
        list_all_samples.append(list_per_sample)

    return pd.DataFrame(list_all_samples, columns=columns_name)


def generate_bm(df, experiment_keys):
    """ Given the flatten DataFrame, containing all the experiment
    here we extract the experiments correspondent to the dictionary experiment_keys.
    :param df: pandas DataFrame containing the flatten df
    :param experiment_keys: dictionary with all the keys.
    :returns df_copy: the reduced dictionary.
    """
    df_copy = df.copy()
    for (k_, v_) in experiment_keys.items():
        df_copy = df_copy[df_copy[k_] == v_]
    return df_copy


def load_all_test_data(data_path):
    """ Here we load the data, given the data path and the indices.
    The flag split denotes if the training dataset is split following a specific
    protocol or if the data are collected all together in a unique matrix.
    :param data_path: the folder or the *.npy. it depends from the flag
    we are loading the entire dataset
    :param class_digit: corresponding data we want to consider

    :returns x the tuple of x[y==class_digit]
    """
    list_file = [f_ for f_ in os.listdir(data_path) if f_.startswith('split')]
    n_splits = len(list_file)

    for j_ in range(n_splits):
        if j_ == 0:
            x = np.load(join(data_path, 'split_%i.npy' % j_))
        else:
            x = np.vstack((x, np.load(join(data_path, 'split_%i.npy' % j_))))
    return x


def cosine_similarity(matrix_1, matrix_2):
    """ Here we pass two activation matrices.
    On the rows there are the number of points,
    each column corresponds to a different hyperplane.
    We return the cosine similarity between activation patterns
    at different epochs
    :param matrix_1: activation for a generic class at a generic epoch
    :param matrix_2: activation for a generic class at a generic epoch
    """
    return [-2 if norm(m1_) * norm(m2_) == 0 else 1-(np.dot(m1_, m2_)/(norm(m1_)*norm(m2_)))
            for (m1_, m2_) in zip(matrix_1.T, matrix_2.T)]


def load_entire_model(weights_file, model_id):
    dim_data = 256
    n_classes = 10
    interesting_mnist = INTERESTING_MNIST_DICT[model_id]

    model = tf.keras.Sequential()
    dim_mnist = 28
    old_kernel_size = 3
    old_pool_size = 2
    new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))
    new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))

    print('new filters size', new_kernel_size)
    print('new max pooling size', new_max_pool)
    model.add(Conv2D(filters=32, kernel_size=(new_kernel_size, new_kernel_size),
                    activation='relu',
                    input_shape=(dim_data, dim_data, 1)))
    model.add(MaxPooling2D((new_max_pool, new_max_pool)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    model.load_weights(weights_file)
    return model


def load_first_layer(entire_model, model_id):
    """
    """
    dim_data = 256
    interesting_mnist = INTERESTING_MNIST_DICT[model_id]
    new_model = tf.keras.Sequential()
    dim_mnist = 28
    old_kernel_size = 3
    old_pool_size = 2
    new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))
    new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))

    print('new filters size', new_kernel_size)
    print('new max pooling size', new_max_pool)

    new_model.add(Conv2D(filters=32,
                         kernel_size=(new_kernel_size, new_kernel_size),
                         kernel_initializer=tf.constant_initializer(entire_model.get_weights()[0]),
                         bias_initializer=tf.constant_initializer(entire_model.get_weights()[1]),
                         activation='relu',
                         input_shape=(dim_data, dim_data, 1)))
    new_model.add(MaxPooling2D((new_max_pool, new_max_pool)))
    return new_model


def load_second_layer(entire_model, model_id):
    dim_data = 256
    interesting_mnist = INTERESTING_MNIST_DICT[model_id]

    model = tf.keras.Sequential()
    dim_mnist = 28
    old_kernel_size = 3
    old_pool_size = 2
    new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))
    new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))

    print('new filters size', new_kernel_size)
    print('new max pooling size', new_max_pool)
    model.add(Conv2D(filters=32,
                     kernel_size=(new_kernel_size, new_kernel_size),
                     kernel_initializer=tf.constant_initializer(entire_model.get_weights()[0]),
                     bias_initializer=tf.constant_initializer(entire_model.get_weights()[1]),
                     activation='relu',
                     input_shape=(dim_data, dim_data, 1)))
    model.add(MaxPooling2D((new_max_pool, new_max_pool)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_initializer=tf.constant_initializer(entire_model.get_weights()[2]),
                     bias_initializer=tf.constant_initializer(entire_model.get_weights()[3]),
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    return model


def load_third_layer(entire_model, model_id):
    dim_data = 256
    interesting_mnist = INTERESTING_MNIST_DICT[model_id]

    model = tf.keras.Sequential()
    dim_mnist = 28
    old_kernel_size = 3
    old_pool_size = 2
    new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))
    new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))

    print('new filters size', new_kernel_size)
    print('new max pooling size', new_max_pool)
    model.add(Conv2D(filters=32,
                     kernel_size=(new_kernel_size, new_kernel_size),
                     kernel_initializer=tf.constant_initializer(entire_model.get_weights()[0]),
                     bias_initializer=tf.constant_initializer(entire_model.get_weights()[1]),
                     activation='relu',
                     input_shape=(dim_data, dim_data, 1)))
    model.add(MaxPooling2D((new_max_pool, new_max_pool)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_initializer=tf.constant_initializer(entire_model.get_weights()[2]),
                     bias_initializer=tf.constant_initializer(entire_model.get_weights()[3]),
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_initializer=tf.constant_initializer(entire_model.get_weights()[4]),
                     bias_initializer=tf.constant_initializer(entire_model.get_weights()[5]),
                     activation='relu'))
    return model


def main():

    path_experiments = "/om/user/vanessad/MNIST_framework/results/MNIST_natural_debug"
    lst_path = ["repetition_%i" % k_ for k_ in range(0, 3)]
    lst_root_path = [join(path_experiments, res_folder_)
                     for res_folder_ in lst_path]

    architecture = '2CNN2FC'
    scenario = 7
    n_train_per_class = 20
    dataset_dimensions = 28

    experiment_keys = dict()
    experiment_keys['architecture'] = architecture
    experiment_keys['scenario'] = scenario
    experiment_keys['n_training'] = n_train_per_class
    experiment_keys['dataset_dimensions'] = dataset_dimensions

    for id_rep_, path_fold_ in enumerate(lst_root_path):
        path_json = join(path_fold_, 'train.json')
        df = flatten_train_json(pd.read_json(path_json))
        if id_rep_ == 0:
            test_accuracy = np.zeros(len(lst_root_path))
        index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
        best_id = search_best_id(path_fold_, index_list)
        ls, ac = np.load(join(path_fold_, 'train_%i/test.npy' % best_id))
        test_accuracy[id_rep_] = ac

    print("MAX ACCURACY")
    print(np.argmax(test_accuracy))
    id_repetition_max_accuracy = np.argmax(test_accuracy)

    exp_path = join(path_experiments,
                    "repetition_%i/train.json" % id_repetition_max_accuracy)
    data_path = '/om/user/vanessad/foveation/mnist_natural/exp_4/dim_28/test'
    save_path = join("/om/user/vanessad/MNIST_framework/runs/activations_mnist_natural",
                     "scenario_%i" % experiment_keys["scenario"])
    os.makedirs(save_path, exist_ok=True)

    test_file_list = [join(data_path, 'split_%i.npy' % k)
                      for k in range(100)]
    root_path = os.path.dirname(exp_path)
    df = flatten_train_json(pd.read_json(exp_path))

    # for each of the n_tr_per class, we look here for the model
    # that shows the best performance, and save its index and weights
    index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
    best_id = search_best_id(root_path, index_list)
    weights_filename = join(root_path, 'train_%i/weights.h5' % best_id)

    print('Experiment corresponding to %i training examples' % n_train_per_class)
    entire_model = load_entire_model(weights_filename, experiment_keys['scenario'])
    model_first_layer = load_first_layer(entire_model, experiment_keys['scenario'])
    model_second_layer = load_second_layer(entire_model, experiment_keys['scenario'])
    model_third_layer = load_third_layer(entire_model, experiment_keys['scenario'])
    max_repr_predict = []
    # one thousand examples per element
    for id_file, test_file in enumerate(test_file_list):
        test_data = np.load(test_file)
        for id_layer, model_ in enumerate([model_first_layer,
                                           model_second_layer,
                                           model_third_layer]):
            repr_predict = model_.predict(test_data.reshape(-1, 256, 256, 1))
            if id_file == 0:
                max_repr_predict.append(np.max(repr_predict, axis=(1, 2)))
                print(len(max_repr_predict))
                print(max_repr_predict[-1].shape)
            else:
                max_repr_predict[id_layer] = np.vstack((max_repr_predict[id_layer],
                                                        np.max(repr_predict, axis=(1, 2))))
    for i in range(3):
        max_repr_predict[i] = np.argsort(max_repr_predict[i], axis=0)[:100]
    pickle.dump(max_repr_predict,
                open(join(save_path,
                          'receptive_field_n_tr_%i' % n_train_per_class,
                          'top_100_list.pkl'), 'wb'))


if __name__ == "__main__":
    main()