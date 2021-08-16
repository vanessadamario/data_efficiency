import json
import os
import numpy as np
from os.path import join


# if you want to include more architecture, add more elements to the dictionary
networks_dict = {'2CNN2FC': [None],
                 'FC': [None]}  # '2CNN2FC_poolAll': [None]
# new learning rate
lr_array = [1e0, 8e-1, 5e-1, 2e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6, 1e-6]
# lr_array = [5e-1, 2e-1, 1e-1, 1e-2, 1e-3, 1e-4] mnist_natural_debug

# /om/user/vanessad/foveation/standardized_MNIST_dataset
datasets_dict = {'standardized_MNIST_dataset': {'dataset_path':  '/om/user/vanessad/foveation/standardized_MNIST_dataset',
                                                'scenario': [1],
                                                'data_dim': [188, 200]}}
# datasets_dict = {'mnist_natural': {'dataset_path': '/om/user/vanessad/foveation/',
#                                    'scenario': [1],
#                                   'data_dim': [188, 200]}}

# 'modified_MNIST_dataset': {'dataset_path': '/om/user/vanessad/foveation/modified_MNIST_dataset',
#                                             'scenario': [1, 2, 3, 4],
#                                            'data_dim': [28, 36, 40, 56, 80]},
# 'standardized_MNIST_dataset': {'dataset_path': '/om/user/vanessad/foveation/standardized_MNIST_dataset',
#                                                 'scenario': [1, 2, 4],
#                                                 'data_dim': [28, 36, 40, 56, 80, 120, 160]}
# 'standardized_MNIST_dataset': {'dataset_path': '/om/user/vanessad/foveation/standardized_MNIST_dataset',
#                                                 'scenario': [1],
#                                                'data_dim': [28, 36, 40, 56, 80, 120, 160]}
# 'mnist_natural': {'dataset_path': '/om/user/vanessad/foveation/mnist_natural',
#                                   'scenario': [4],
#                                   'data_dim': [28, 80, 150]}
# 'mnist_natural': {'dataset_path': '/om/user/vanessad/foveation/mnist_natural',
#                                    'scenario': [3],
#                                    'data_dim': [28, 80, 150, 200]}

# n_array = [20, 50, 100, 200, 300, 500, 1000]
# n_array = [1, 5, 20, 50, 100]
n_array = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
# batch_size_small = [1, 2, 5, 8, 10]
batch_size_list = [10, 32, 50, 100]


class Hyperparameters(object):
    """ Add hyper-parameters in init so when you read a json, it will get updated as your latest code. """
    def __init__(self,
                 learning_rate=5e-2,
                 architecture=None,
                 epochs=500,
                 batch_size=32,
                 optimizer='sgd',
                 lr_at_plateau=True,
                 reduction_factor=None,
                 validation_check=True):
        """
        :param learning_rate: float, the initial value for the learning rate.
        :param architecture: str, the architecture types.
        :param epochs: int, the number of epochs we want to train.
        :param batch_size: int, the dimension of the batch size.
        :param optimizer: str, the optimizer type.
        :param lr_at_plateau: bool, protocol to decrease the learning rate.
        :param reduction_factor, int, the factor which we use to reduce the learning rate.
        :param validation_check: bool, if we want to keep track of validation loss as a stopping criterion.
        """
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_at_plateau = lr_at_plateau
        self.reduction_factor = reduction_factor
        self.validation_check = validation_check


class Dataset(object):
    """ Here we save the dataset specific related to each experiment. The name of the dataset,
    the scenario, if we modify the original dataset, and the dimensions of the input.
    This is valid for the modified_MNIST_dataset, verify if it is going to be valid next"""
    def __init__(self,
                 dataset_name='modified_MNIST_dataset',
                 dataset_path='/om/user/vanessad/foveation/modified_MNIST_dataset',
                 scenario=1,
                 dataset_dimensions=80,
                 n_training=10):
        """
        :param dataset_name: str, name of the folder of the experiments
        :param dataset_path: str, dataset path in the server
        :param scenario: str, the learning paradigm, foveation, noise, different backgrounds
        :param dataset_dimensions: str, dimensionality of the dataset, as we play with this parameter
        :param n_training: int, number of training examples
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.scenario = scenario
        self.dataset_dimensions = dataset_dimensions
        self.n_training = n_training


class Experiment(object):
    """
    This class represents your experiment. It includes all the classes above and some general information about the
    experiment index.
    IF YOU ADD ANOTHER CLASS, MAKE SURE TO INCLUDE IT HERE.
    """
    def __init__(self, id, output_path, train_completed=False, hyper=None, dataset=None):
        # TODO: it depends on mnist, 10 is the number of classes
        """
        :param id: index of output data folder
        :param output_path: output directory
        :param complete_path: path where to save the *.txt which denote if the exp is completed
        :param train_completed: bool, it indicates if the experiment has already been trained
        :param hyper: instance of Hyperparameters class
        :param dataset: instance of Dataset class
        """
        if hyper is None:
            hyper = Hyperparameters()
        if dataset is None:
            dataset = Dataset()

        self.id = id
        self.output_path = output_path
        self.train_completed = train_completed
        self.hyper = hyper
        self.dataset = dataset

        if self.dataset.n_training < 3:
            self.hyper.batch_size = 10


def decode_exp(dct):
    """ When reading a json file, it is originally a dictionary
    which is hard to work with in other parts of the code.
    IF YOU ADD ANOTHER CLASS TO EXPERIMENT, MAKE SURE TO INCLUDE IT HERE.
    This function goes through the dictionary and turns it into an instance of Experiment class.
        :parameter dct: dictionary of parameters as saved in the *.json file.
        :returns exp: instance of the Experiment class.
    """
    hyper = Hyperparameters()
    for key in hyper.__dict__.keys():
        if key in dct['hyper'].keys():
            hyper.__setattr__(key, dct['hyper'][key])
    dataset = Dataset()
    for key in dataset.__dict__.keys():
        if key in dct['dataset'].keys():
            dataset.__setattr__(key, dct['dataset'][key])

    exp = Experiment(dct['id'], dct['output_path'], dct['train_completed'], hyper, dataset)
    return exp


def exp_exists(exp, info):
    """ This function checks if the experiment exists in your json file to avoid duplicate experiments.
    """
    # TODO: is this function called in other parts, except from generate_experiments?
    # do we want to put also the flag train_completed here, correct?
    dict_new = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
    dict_new_wo_id = {i: dict_new[i]
                      for i in dict_new if (i != 'id' and i != 'output_path' and i != 'train_completed')}
    for idx in info:
        dict_old = info[idx]
        dict_old_wo_id = {i: dict_old[i]
                          for i in dict_old if (i != 'id' and i != 'output_path' and i != 'train_completed')}
        if dict_old_wo_id == dict_new_wo_id:
            return idx
    return False


def generate_experiments(output_path):
    """ This function is called to make your train.json file or append to it.
    You should change the loops for your own usage.
    The info variable is a dictionary that first reads the json file if there exists any,
    appends your new experiments to it, and dumps it into the json file again
    """
    # TODO: include the dataset path, generate everything from there
    # WARNING: if the *.json is empty it complains
    info = {}

    info_path = output_path + 'train.json'
    dirname = os.path.dirname(info_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0
    elif os.path.isfile(info_path):
        with open(info_path) as infile:
            info = json.load(infile)
            if info:  # it is not empty
                idx_base = int(list(info.keys())[-1]) + 1  # concatenate
            else:
                idx_base = 0
    else:
        idx_base = 0

    # These loops indicate your experiments. Change them accordingly.
    for datum_dict in datasets_dict.keys():  # for each dataset, here we have one
        for net_ in networks_dict.keys():
            for spec_scenario in datasets_dict[datum_dict]['scenario']:  # for each element of the dictionary
                for spec_paradigm in datasets_dict[datum_dict]['data_dim']:
                    for lr_ in lr_array:
                        for n_ in n_array:
                            batch_ = [10] if n_ <= 2 else batch_size_list
                            for bs_ in batch_:
                                hyper = Hyperparameters(learning_rate=lr_,
                                                        architecture=net_,
                                                        batch_size=bs_)
                                dataset = Dataset(dataset_name=datum_dict,
                                                  dataset_path=datasets_dict[datum_dict]['dataset_path'],
                                                  scenario=spec_scenario,
                                                  dataset_dimensions=spec_paradigm,
                                                  n_training=n_)
                                exp = Experiment(id=idx_base,
                                                 output_path=output_path+'train_'+str(idx_base),
                                                 train_completed=False,
                                                 hyper=hyper,
                                                 dataset=dataset)
                                idx = exp_exists(exp, info)
                                if idx is not False:
                                    print("The experiment already exists with id:", idx)
                                    continue
                                s = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
                                print(s)
                                info[str(idx_base)] = s
                                idx_base += 1

        with open(info_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)


def get_experiment(output_path, id):
    """
    This function is called when you want to get the details of your experiment
    given the index (id) and the path to train.json
    """
    info_path = join(output_path, 'train.json')
    with open(info_path) as infile:
        trained = json.load(infile)
    opt = trained[str(id)]  # access to the experiment details through the ID
    exp = decode_exp(opt)   # return an Experiment object

    print('Retrieved experiment:')
    for key in exp.__dict__.keys():
        if key is 'hyper':  # hyper-parameters details
            print('hyper:', exp.hyper.__dict__)
        elif key is 'dataset':  # dataset details
            print('dataset: ', exp.dataset.__dict__)
        else:
            print(key, ':', exp.__getattribute__(key))

    return exp
