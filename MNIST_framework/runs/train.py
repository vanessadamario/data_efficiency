import os
import json
import pandas as pd
from os.path import join
from runs.network import Network


def train_network(opt, check_train=False):
    net = Network(opt)
    net.run(check_train)
    net.eval_metrics()
    net.save_outputs()
    return net


def check_and_train(opt, output_path, check_train):
    """ Check if the experiments has already been performed.
    If it is not, train otherwise retrieve the path relative to the experiment.
    :param opt: Experiment instance. It contains the output path for the experiment
    under study
    :param output_path: the output path for the *.json file. necessary to change the *.json
    :param check_train: if check_train is True, we save the dimensionality of the network
    at different time steps
    """
    if opt.train_completed:
        print("Object: ", opt)
        print("Experiment already trained in " + opt.output_path)
        return

    print(output_path)
    print('check and train', opt.output_path)

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    train_network(opt, check_train)

    # we write an empty *.txt file with the completed experiment
    flag_completed_dir = join(output_path, 'flag_completed')
    os.makedirs(flag_completed_dir, exist_ok=True)
    file_object = open(join(flag_completed_dir, "complete_%s.txt" % str(opt.id)), "w")
    file_object.close()

