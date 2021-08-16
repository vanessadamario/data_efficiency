import os
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
import sys
from runs.utils import _generate_index_tr_vl
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def load_data(data_path, index, flag_split=False, y=None):
    """ Here we load the data, given the data path and the indices.
    The flag split denotes if the training dataset is split following a specific
    protocol or if the data are collected all together in a unique matrix.
    :param data_path: the folder or the *.npy. it depends from the flag
    :param index: the index of the elements we want to load, if index is the same size of y
    we are loading the entire dataset
    :param flag_split: the bool flag to see if there is splitting or not
    :param y: the array of labels

    :returns (x, y) the tuple of x[index], y[index]
    """
    if flag_split:
        n_total = y.size  # total number of points from which to extract

        if index.size < y.size:
            list_file = [f_ for f_ in os.listdir(data_path) if f_.startswith('split')]
            n_splits = len(list_file)
            split_size = n_total // n_splits  # for the whole set of data
            index = np.sort(index)
            index_grouped_by_split = [[index[np.logical_and(index >= split_size * i_,
                                                            index < split_size * (i_+1))],
                                       i_]
                                      for i_ in range(n_splits)]

            flag_first_element = True
            for j_, (id_indices, id_iteration) in enumerate(index_grouped_by_split):

                print('element')
                sys.stdout.flush()
                print(j_)
                sys.stdout.flush()

                if len(id_indices) > 0:
                    if flag_first_element:
                        x = np.load(join(data_path, 'split_%i.npy' % j_))[(id_indices % split_size)]
                        flag_first_element = False
                    else:
                        x = np.vstack((x,
                                       np.load(join(data_path, 'split_%i.npy' % j_))[(id_indices % split_size)]))
            return x, y[index]

        elif index.size == n_total:
            list_file = [f_ for f_ in os.listdir(data_path) if f_.startswith('split')]
            n_splits = len(list_file)
            for j_ in range(n_splits):
                if j_ == 0:
                    x = np.load(join(data_path, 'split_%i.npy' % j_))
                else:
                    x = np.vstack((x, np.load(join(data_path, 'split_%i.npy' % j_))))
            return x, y

    else:
        return np.load(data_path)[index], y[index]


class Network(tf.keras.Sequential):
    """ We generate here the class Network, which contains the different NN
    models implemented in the experiments. This class inherits from the tf
    Model class. Here we will just implement the different architectures.
    By default, as we generate the Network instance, we build the network. """
    def __init__(self, opt, check_train=False):
        """
        :param opt: Experiment instance
        :param check_train: boolean flag that save the dimensions if true
        and the weights across training
        """
        super(Network, self).__init__()
        self.opt = opt
        self.history = None
        self.fitted = False
        self.test_loss = None
        self.test_accuracy = None
        self.flag_split_dataset = None
        # TODO: make it general. it works because we are dealing with square images (MNIST)
        # TODO: we fix the number of classes at 10 because we are dealing with MNIST -- change
        self.dim_data = None

        given_pool_size = False
        n_classes = 10

        if self.opt.dataset.scenario not in [3, 5, 6, 7, 8, 9, 10]:
            # for these cases we change the receptive field size but
            # we do not change the dataset
            self.load_dataset_name = self.opt.dataset.scenario
        else:
            self.load_dataset_name = 4

        if self.opt.dataset.dataset_name == 'modified_MNIST_dataset':
            self.flag_split_dataset = False
            min_delta_early_stop = 1e-6
        elif self.opt.dataset.dataset_name == 'standardized_MNIST_dataset':
            self.flag_split_dataset = True
            min_delta_early_stop = 1e-6
        elif self.opt.dataset.dataset_name == 'mnist_natural':
            self.flag_split_dataset = True
            min_delta_early_stop = 1e-6  # 0,1,2,3 was 1e-4
        elif self.opt.dataset.dataset_name == 'mnist_natural_segmented':
            self.flag_split_dataset = True
            min_delta_early_stop = 1e-6

        else:
            raise ValueError('The dataset provided does not exists')
        if opt.dataset.scenario == 1:
            self.dim_data = opt.dataset.dataset_dimensions
            interesting_mnist = 28
        elif opt.dataset.scenario == 2:
            self.dim_data = opt.dataset.dataset_dimensions
            interesting_mnist = opt.dataset.dataset_dimensions
        elif opt.dataset.scenario == 3:
            self.dim_data = 256  # it works for mnist_natural only
            interesting_mnist = 256
        # if opt.dataset.scenario < 3: # TODO: remove old version
        #     self.dim_data = opt.dataset.dataset_dimensions
        elif opt.dataset.scenario == 4:
            interesting_mnist = opt.dataset.dataset_dimensions
            if self.opt.dataset.dataset_name == 'modified_MNIST_dataset':
                self.dim_data = 150
            elif self.opt.dataset.dataset_name == 'standardized_MNIST_dataset':
                self.dim_data = 200
            elif self.opt.dataset.dataset_name == 'mnist_natural':
                self.dim_data = 256
            elif self.opt.dataset.dataset_name == 'mnist_natural_segmented':
                self.dim_data = 256
        elif opt.dataset.scenario == 5:
            # receptive field fixed an half smaller
            self.dim_data = 256  # it works for mnist_natural only
            interesting_mnist = 128
        elif opt.dataset.scenario == 6:
            # receptive field fixed a quarter smaller
            self.dim_data = 256  # it works for mnist_natural only
            interesting_mnist = 64
        elif opt.dataset.scenario == 7:
            # receptive field fixed a quarter smaller
            self.dim_data = 256  # it works for mnist_natural only
            interesting_mnist = 28
        elif opt.dataset.scenario == 8:
            self.dim_data = 256
            interesting_mnist = 128
            given_pool_size = True
            pool_size = 1
        elif opt.dataset.scenario == 9:
            self.dim_data = 256
            given_pool_size = True
            pool_size = 9
            interesting_mnist = 128
        elif opt.dataset.scenario == 10:
            self.dim_data = 256
            interesting_mnist = 128

        if self.opt.hyper.architecture == 'FC':
            nodes = 128
            self.add(Flatten(input_shape=(self.dim_data, self.dim_data)))
            self.add(Dense(nodes, activation='relu'))
            self.add(Dense(n_classes, activation='softmax'))

        elif self.opt.hyper.architecture == '2CNN2FC':
            if not self.flag_split_dataset:
                self.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                                input_shape=(self.dim_data, self.dim_data, 1)))
                self.add(MaxPooling2D((2, 2)))
            else:
                dim_mnist = 28
                old_kernel_size = 3
                old_pool_size = 2
                new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))

                if given_pool_size:
                    new_max_pool = pool_size
                else:
                    new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))

                print('new filters size', new_kernel_size)
                print('new max pooling size', new_max_pool)
                # new_kernel_size = int(np.round(self.opt.dataset.dataset_dimensions / dim_mnist * old_kernel_size))
                # new_max_pool = int(np.round(self.opt.dataset.dataset_dimensions / dim_mnist * old_pool_size))

                self.add(Conv2D(filters=32, kernel_size=(new_kernel_size, new_kernel_size),
                                activation='relu',
                                input_shape=(self.dim_data, self.dim_data, 1)))
                self.add(MaxPooling2D((new_max_pool, new_max_pool)))
            self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            self.add(MaxPooling2D((2, 2)))
            self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            self.add(Flatten())
            self.add(Dense(64, activation='relu'))
            self.add(Dense(n_classes, activation='softmax'))

        elif self.opt.hyper.architecture == '2CNN2FC_poolAll':
            if not self.flag_split_dataset:
                self.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                                input_shape=(self.dim_data, self.dim_data, 1)))
                self.add(MaxPooling2D((2, 2)))
            else:
                dim_mnist = 28
                old_kernel_size = 3
                old_pool_size = 2
                new_kernel_size = int(np.round(interesting_mnist / dim_mnist * old_kernel_size))
                new_max_pool = int(np.round(interesting_mnist / dim_mnist * old_pool_size))
                print('new filters size', new_kernel_size)
                print('new max pooling size', new_max_pool)
                # new_kernel_size = int(np.round(self.opt.dataset.dataset_dimensions / dim_mnist * old_kernel_size))
                # new_max_pool = int(np.round(self.opt.dataset.dataset_dimensions / dim_mnist * old_pool_size))

                self.add(Conv2D(filters=32, kernel_size=(new_kernel_size, new_kernel_size),
                                activation='relu',
                                input_shape=(self.dim_data, self.dim_data, 1)))
                self.add(MaxPooling2D((new_max_pool, new_max_pool)))
            self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            self.add(MaxPooling2D((2, 2)))
            self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            self.add(GlobalAveragePooling2D())  # avg pooling
            self.add(Dense(64, activation='relu'))
            self.add(Dense(n_classes, activation='softmax'))

        if self.opt.hyper.optimizer == 'sgd':
            sgd = SGD(lr=self.opt.hyper.learning_rate, momentum=0., nesterov=False)
        else:
            raise ValueError('This optimizer has not been included yet')

        if self.opt.hyper.lr_at_plateau:
            self._lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.1,
                                                   patience=5,
                                                   min_lr=0)
        self._early_stopping = EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             min_delta=min_delta_early_stop)

        self.compile(optimizer=sgd,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    def run(self, check_train=False):
        """ Here we train the algorithm.
        :param check_train: if True, we save the activations at each epoch.
        """
        (_, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        idx_tr, idx_vl = _generate_index_tr_vl(y_train, self.opt.dataset.n_training, n_vl=1000)

        if self.flag_split_dataset:
            x_tr, y_tr = load_data(join(self.opt.dataset.dataset_path,
                                        'exp_%i' % self.load_dataset_name,
                                        'dim_%i' % self.opt.dataset.dataset_dimensions,
                                        'train'),
                                   idx_tr, flag_split=True, y=y_train)
            x_vl, y_vl = load_data(join(self.opt.dataset.dataset_path,
                                        'exp_%i' % self.load_dataset_name,
                                        'dim_%i' % self.opt.dataset.dataset_dimensions,
                                        'train'),
                                   idx_vl, flag_split=True, y=y_train)

        else:
            x_tr, y_tr = load_data(join(self.opt.dataset.dataset_path,
                                     'exp_%i_dim_%i_tr.npy' % (self.load_dataset_name,
                                                               self.opt.dataset.dataset_dimensions)),
                                   idx_tr, y=y_train)
            x_vl, y_vl = load_data(join(self.opt.dataset.dataset_path,
                                     'exp_%i_dim_%i_tr.npy' % (self.load_dataset_name,
                                                               self.opt.dataset.dataset_dimensions)),
                                   idx_vl, y=y_train)

        if self.opt.hyper.architecture.startswith('2CNN2FC'):
            print("reshape")
            sys.stdout.flush()
            _, dim1, dim2 = x_tr.shape
            x_tr = x_tr.reshape(-1, dim1, dim2, 1)
            x_vl = x_vl.reshape(-1, dim1, dim2, 1)

        if check_train:
            np.save(join(self.opt.output_path, 'tr_indices.npy'), idx_tr)
            np.save(join(self.opt.output_path, 'vl_indices.npy'), idx_tr)
            checkpoint_path = join(self.opt.output_path, "checkpoint/cp_{epoch:02d}.ckpt")
            checkpoint_dir = os.path.dirname(checkpoint_path)
            os.makedirs(checkpoint_dir, exist_ok=True)

            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          save_best_only=False,
                                          verbose=1,
                                          save_freq='epoch')
            callbacks_list = [self._lr_reduction,
                              self._early_stopping,
                              cp_callback]
        else:
            callbacks_list = [self._lr_reduction,
                              self._early_stopping]

        history = self.fit(x=x_tr,
                           y=y_tr,
                           epochs=self.opt.hyper.epochs,
                           batch_size=self.opt.hyper.batch_size,
                           validation_data=(x_vl, y_vl),
                           callbacks=callbacks_list)
        print(self.summary())
        print('fitted data, end of run function')
        sys.stdout.flush()
        self.history = history.history
        self.fitted = True

        if check_train and self.opt.hyper.architecture == 'FC':
            # here the fun begins
            nodes = 128
            n_classes = 10

            list_cp_files = [f_ for f_ in os.listdir(join(self.opt.output_path, 'checkpoint'))
                             if f_.startswith('cp_')]
            n_epochs = np.array([int(k_.split('.')[0].split('_')[-1])
                                 for k_ in list_cp_files])

            activations_overall = np.zeros((self.opt.hyper.epochs,
                                            n_classes,
                                            self.opt.dataset.n_training,
                                            nodes))

            for id_e_, e_ in enumerate(range(np.min(n_epochs), np.max(n_epochs)+1)):
                new_model = tf.keras.Sequential()
                new_model.add(Flatten(input_shape=(self.dim_data, self.dim_data,)))
                new_model.add(Dense(nodes, activation='relu'))
                new_model.add(Dense(n_classes, activation='softmax'))
                new_model.compile(optimizer='sgd',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])
                print(new_model.summary())
                print(join(self.opt.output_path, f'checkpoint/cp_{e_:02d}.ckpt'))
                new_model.load_weights(join(self.opt.output_path,
                                            f'checkpoint/cp_{e_:02d}.ckpt'))
                weights, bias = new_model.layers[1].get_weights()
                activations = np.maximum(np.dot(x_tr.reshape(-1, self.dim_data * self.dim_data),
                                                weights) + bias, 0)

                for y_ in np.unique(y_tr):
                    activations_overall[id_e_, y_, :, :] = activations[y_tr == y_]
                    # here we need to check the activations for each class
                    # recover the ys
                    # smarter way to do it
            np.save(join(self.opt.output_path, 'activations.npy'), activations_overall)

        return self

    def eval_metrics(self):
        """ Here we evaluate the model performance on the test set.
        """
        if not self.fitted:
            raise ValueError('The model has not been fitted yet')

        (_, _), (_, y_test) = tf.keras.datasets.mnist.load_data()
        del _
        n_ts_samples = y_test.size
        print('I must load test data')
        sys.stdout.flush()

        if self.flag_split_dataset:
            test_loss_lst = []
            test_accuracy_lst = []
            # we evaluate in batches because the dataset is too big to fit in memory
            data_path = join(self.opt.dataset.dataset_path,
                             'exp_%i' % self.load_dataset_name,
                             'dim_%i' % self.opt.dataset.dataset_dimensions,
                             'test')

            list_file = [f_ for f_ in os.listdir(data_path) if f_.startswith('split')]
            n_splits = len(list_file)
            el_per_split = n_ts_samples // n_splits
            for j_ in range(n_splits):
                x_ts = np.load(join(data_path, 'split_%i.npy' % j_))
                start_ = j_*el_per_split
                if (j_+1)*el_per_split > n_ts_samples:
                    end_ = n_ts_samples
                else:
                    end_ = (j_+1)*el_per_split
                y_ts = y_test[start_:end_]

                if self.opt.hyper.architecture.startswith('2CNN2FC'):
                    _, dim1, dim2 = x_ts.shape
                    x_ts = x_ts.reshape(-1, dim1, dim2, 1)

                print('test data loaded, split %i' %j_)
                sys.stdout.flush()
                tmp_test_loss, tmp_test_accuracy = self.evaluate(x_ts, y_ts, verbose=2)

                test_loss_lst.append(tmp_test_loss)
                test_accuracy_lst.append(tmp_test_accuracy)
            test_loss = np.mean(np.array(test_loss_lst))
            test_accuracy = np.mean(np.array(test_accuracy_lst))
            self.test_loss = test_loss
            self.test_accuracy = test_accuracy
            print('eval test data, end of eval_metrics function')
            sys.stdout.flush()

            return test_loss, test_accuracy

        else:
            x_ts, y_ts = load_data(join(self.opt.dataset.dataset_path,
                                        'exp_%i_dim_%i_ts.npy' % (self.load_dataset_name,
                                                                  self.opt.dataset.dataset_dimensions)),
                                   index=np.arange(y_test.size), y=y_test)
        print('test data loaded, here we are')
        sys.stdout.flush()

        if self.opt.hyper.architecture.startswith('2CNN2FC'):
            print("reshape")
            sys.stdout.flush()
            _, dim1, dim2 = x_ts.shape
            x_ts = x_ts.reshape(-1, dim1, dim2, 1)

        test_loss, test_accuracy = self.evaluate(x_ts, y_ts, verbose=2)
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        print('eval test data, end of eval_metrics function')
        sys.stdout.flush()

        sys.stdout.flush()
        return test_loss, test_accuracy

    def save_outputs(self):
        """ Save the content of the network.
        We save the weights and the history.
        We do not save the object Network, because it is redundant. """
        self.save_weights(join(self.opt.output_path, 'weights.h5'),
                          save_format='h5')
        print("saving weights")
        sys.stdout.flush()
        df = pd.DataFrame(data=self.history)
        df.to_csv(join(self.opt.output_path, 'history.csv'))
        np.save(join(self.opt.output_path, 'test.npy'), np.append(self.test_loss,
                                                                  self.test_accuracy))
        del df





