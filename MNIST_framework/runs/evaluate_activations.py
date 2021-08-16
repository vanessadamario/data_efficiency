# in this script we plot the average activation of neurons, in particular
# how much these change based on the initial value
import os
import pandas as pd
import numpy as np
from os.path import join
from numpy.linalg import norm
import matplotlib.pyplot as plt
from search_in_json import search_best_id, flatten_train_json, generate_bm

# do we want to import some scikit-learn routine to evaluate
# the among points of the same class or from different classes?


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


def main():
    path = '/om/user/vanessad/MNIST_framework/results_dynamics/train.json'
    path_figures = '/om/user/vanessad/MNIST_framework/viz_results_dynamics'

    dirname = os.path.dirname(path)
    df = flatten_train_json(pd.read_json(path))
    n_classes = 10

    dataset = 'standardized_MNIST_dataset'
    architecture = 'FC'
    experiment_keys = {'dataset_name': dataset,
                       'architecture': architecture,
                       'epochs': 50}

    # PLOT FOR COSINE SIMILARITY
    scenario_list = [1, 2, 4]
    for sc in scenario_list:
        experiment_keys['scenario'] = sc
        output_path = join(path_figures, dataset, architecture, 'scenario_%i' % sc)
        os.makedirs(output_path, exist_ok=True)

        for n_tr in sorted(list(set(df.n_training))):
            experiment_keys['n_training'] = n_tr
            if n_tr < 10:
                experiment_keys['batch_size'] = 10
            else:
                experiment_keys['batch_size'] = 32

            for dim in sorted(list(set(df.dataset_dimensions))):
                experiment_keys['dataset_dimensions'] = dim
                index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                best_id = search_best_id(dirname, index_list)
                path_best_id = join(dirname, 'train_%i' % best_id)
                history = pd.read_csv(join(path_best_id, 'history.csv'))
                activations = np.load(join(path_best_id, 'activations.npy'))

                fig, ax = plt.subplots(figsize=(20, 10), ncols=n_classes // 2, nrows=2)
                for k__ in range(n_classes):
                    col = int(k__ % (n_classes / 2))
                    row = int(k__ / (n_classes / 2))
                    similarity_vals = np.array(cosine_similarity(activations[0, k__],
                                                                 activations[history.shape[0] - 1, k__]))
                    id_active = np.array(np.squeeze(np.argwhere(np.array(similarity_vals) != -2)))
                    n_inactive = np.argwhere(np.array(similarity_vals) == -2).size

                    similarity_vals = similarity_vals[id_active]
                    ax[row, col].hist(similarity_vals,
                                      bins=20,
                                      alpha=0.5,
                                      label='inactive: %i' % n_inactive)

                    ax[row, col].set_title('class %i' % k__)
                    ax[row, col].set_xlim([-1.01, 1.01])
                    ax[row, col].set_ylim([0, 128])

                    ax[row, 0].set_ylabel('# units, total 128')
                    ax[row, col].legend()
                plt.savefig(join(output_path, 'n_train_%i_dim_%i.pdf' % (n_tr, dim)))
                plt.close()


if __name__ == "__main__":
    main()