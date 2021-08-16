# output_dimension = [28, 36, 40, 56, 80, 120, 160]
# scenario = [1, 2, 4]

import os
import numpy as np
import argparse
from os.path import join
from tensorflow.keras.datasets import mnist
from DATASET_GENERATOR import DatasetGenerator

root_path = '/om/user/vanessad/foveation'

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=int, required=True)
parser.add_argument('--output_dimension', type=int, required=True)
parser.add_argument('--dataset_name', type=int, required=False)

FLAGS = parser.parse_args()

print(FLAGS.output_dimension)
print(FLAGS.scenario)

if FLAGS.dataset_name is None:
    FLAGS.dataset_name = 'standardized_MNIST_dataset'

n_splits = 100
folder_dataset = join(root_path, FLAGS.dataset_name)
os.makedirs(folder_dataset, exist_ok=True)

folder_scenario = join(folder_dataset, 'exp_%i' % FLAGS.scenario)
os.makedirs(folder_scenario, exist_ok=True)

folder_dimension = join(folder_scenario, 'dim_%i' % FLAGS.output_dimension)
os.makedirs(folder_dimension)

folder_train = join(folder_dimension, 'train')
folder_test = join(folder_dimension, 'test')

os.makedirs(folder_train)
os.makedirs(folder_test)


if FLAGS.dataset_name == 'standardized_MNIST_dataset':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization
    list_of_tr_splits = np.split(x_train, n_splits, axis=0)
    list_of_ts_splits = np.split(x_test, n_splits, axis=0)

elif FLAGS.dataset_name == 'foveated_MNIST_dataset':
    path_to_std_mnist = join(root_path, 'standardized_MNIST_dataset',
                             'exp_%i' % FLAGS.scenario,
                             'dim_%i' % FLAGS.output_dimension)
    list_of_tr_splits = [np.load(join(path_to_std_mnist, 'train', ' split_%i.npy' % j)
                                 for j in range(n_splits))]
    list_of_ts_splits = [np.load(join(path_to_std_mnist, 'test', 'split_%i.npy' % j)
                                 for j in range(n_splits))]

else:
    raise ValueError("The required dataset still does not exists")


for kk, x_train_split in enumerate(list_of_tr_splits):
    DG = DatasetGenerator(x_train_split,
                          output_dim=FLAGS.output_dimension,
                          scenario=FLAGS.scenario)
    DG.run()
    np.save(join(folder_train, 'split_%i.npy' % kk), DG.output)

for kk, x_test_split in enumerate(list_of_ts_splits):
    DG = DatasetGenerator(x_test_split,
                          output_dim=FLAGS.output_dimension,
                          scenario=FLAGS.scenario)
    DG.run()
    np.save(join(folder_test, 'split_%i.npy' % kk), DG.output)