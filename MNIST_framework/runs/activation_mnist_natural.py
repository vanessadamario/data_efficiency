# in this script we plot the average activation of neurons, in particular
# how much these change based on the initial value

import os
from numpy.linalg import norm
from os.path import join
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalMaxPool2D

import sys
sys.path.append("..")

from runs.mean_image_torralba_mnist_natural import INTERESTING_MNIST_DICT
from runs.mean_image_torralba_mnist_natural import search_best_id, flatten_train_json, generate_bm, load_all_test_data
from runs.mean_image_torralba_mnist_natural import load_entire_model, load_first_layer, load_second_layer, load_third_layer

FILTER_SIZE_DICT = {3: 27,
                    5: 14,
                    7: 3}
MAX_POOL_SIZE_DICT = {3: 18,
                      5: 9,
                      7: 2}
IMAGE_SIZE = 256


def generate_bm_center(model_id, image_size):
    original_filter_size = 3
    original_max_pooling = 2
    layer_1_rep_dim = int(np.floor((image_size - (FILTER_SIZE_DICT[model_id]-1)) /
                                   MAX_POOL_SIZE_DICT[model_id]))
    layer_2_rep_dim = int(np.floor((layer_1_rep_dim - (original_filter_size-1)) /
                                   original_max_pooling))
    layer_3_rep_dim = layer_2_rep_dim - (original_filter_size-1)
    return layer_1_rep_dim, layer_2_rep_dim, layer_3_rep_dim


def concentric(dim_rep):
    bool_mask_list = []
    for k_ in range((dim_rep + 1) // 2):
        z = np.zeros((dim_rep, dim_rep))
        xx, yy = np.meshgrid(np.arange(k_, dim_rep - k_),
                             np.arange(k_, dim_rep - k_))
        z[xx, yy] = 1
        xx_, yy_ = np.meshgrid(np.arange(k_ + 1, dim_rep - k_ - 1),
                               np.arange(k_ + 1, dim_rep - k_ - 1))
        z[xx_, yy_] = 0
        bool_mask_list.append(z)
    return bool_mask_list


def main():  # eval activation

    scenario_list = [5, 7]
    n_train_per_class_list = [20, 1000]
    path_experiments = "/om/user/vanessad/MNIST_framework/results/MNIST_natural_debug"
    lst_path = ["repetition_%i" % k_ for k_ in range(0, 3)]
    lst_root_path = [join(path_experiments, res_folder_)
                     for res_folder_ in lst_path]
    dataset_dimensions = 28
    data_path = '/om/user/vanessad/foveation/mnist_natural/exp_4/dim_28/test'
    root_path_figure = '/om/user/vanessad/MNIST_framework/runs/activation_maps'

    architecture = '2CNN2FC'

    experiment_keys = dict()
    experiment_keys['architecture'] = architecture
    experiment_keys['dataset_dimensions'] = dataset_dimensions

    for scenario in scenario_list:
        for n_train_per_class in n_train_per_class_list:

            l1, l2, l3 = generate_bm_center(scenario, IMAGE_SIZE)
            bm_list_layer1 = concentric(l1)
            bm_list_layer2 = concentric(l2)
            bm_list_layer3 = concentric(l3)

            experiment_keys['scenario'] = scenario
            experiment_keys['n_training'] = n_train_per_class

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
            save_path = join("/om/user/vanessad/MNIST_framework/runs/activations_mnist_natural",
                             "scenario_%i" % experiment_keys["scenario"])
            os.makedirs(save_path, exist_ok=True)

            root_path = os.path.dirname(exp_path)
            df = flatten_train_json(pd.read_json(exp_path))

            # for each of the n_tr_per class, we look here for the model
            # that shows the best performance, and save its index and weights
            index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
            best_id = search_best_id(root_path, index_list)
            weights_filename = join(root_path, 'train_%i/weights.h5' % best_id)

            # we consider the first 1k examples from the test set
            test_data = np.zeros((1000, 256, 256))
            for k_ in range(10):
                test_data[k_*100:(k_+1)*100] = np.load(join(data_path, "split_%i.npy" % k_))
            print("LOADED TEST DATA")

            # the three are for different amount of training examples
            # lists for activations in the first layer
            # all these evaluations overestimate the contribution of the background
            # we include spurious correlation from the background in the
            # to compute the activations in the center
            path_figure = join(root_path_figure,
                               "scenario_%i" % scenario,
                               "n_train_%i" % n_train_per_class)

            model_ = load_entire_model(weights_filename, scenario)

            for j_, (load_func_, b_mask_) in enumerate(zip([load_first_layer,
                                                            load_second_layer,
                                                            load_third_layer],
                                                           [bm_list_layer1,
                                                            bm_list_layer2,
                                                            bm_list_layer3])):

                print(len(b_mask_))
                print(b_mask_[0].shape, b_mask_[1].shape)
                model_at_layer = load_func_(model_, scenario)
                repr_predict = model_at_layer.predict(test_data.reshape(-1, 256, 256, 1))
                print(repr_predict.shape)
                norm_per_image = np.max(repr_predict, axis=(1, 2, 3))
                max_activation_array = np.zeros((len(b_mask_), repr_predict.shape[0]))

                for id_bm, bm_concentric in enumerate(b_mask_):
                    concentric_vals = repr_predict[:, bm_concentric.astype(bool)]
                    # this returns (# n_points, unrolled_boolean_mask)
                    print(concentric_vals.shape)
                    max_activation = np.max(concentric_vals, axis=(1, 2)) / norm_per_image
                    max_activation_array[id_bm, :] = max_activation
                np.save(join(path_figure, "normalized_max_activation_layer_%i.npy" % (j_+1)),
                        max_activation_array)


if __name__ == "__main__":
    main()


""" old boolean mask for scenario 3
    # we generate the boolean mask for
    # center and background
    bm_1st = np.zeros((12, 12))
    bm_1st[6, 6] = 1
    bm_1st[5, 6] = 1
    bm_1st[6, 5] = 1
    bm_1st[6, 7] = 1
    bm_1st[7, 6] = 1
    bm_1st = bm_1st.astype(bool)

    bm_2nd = np.zeros((5, 5))
    bm_2nd[2, 2] = 1
    bm_2nd[1, 2] = 1
    bm_2nd[2, 1] = 1
    bm_2nd[2, 3] = 1
    bm_2nd[3, 2] = 1
    bm_2nd = bm_2nd.astype(bool)

    bm_3rd = np.zeros((3, 3))
    bm_3rd[1, 1] = 1
    bm_3rd = bm_3rd.astype(bool)



    max_repr_predict = np.max(repr_predict, axis=(1, 2))
    # highest activation for every image and every filter
    const_each_filter = np.max(max_repr_predict, axis=0)
    # highest activation each filter
    id_max_filter = np.argmax(max_repr_predict, axis=-1)
    # for each image, which filter gives the highest activation
    max_feature_map_sample = np.zeros((repr_predict.shape[0], repr_predict.shape[1], repr_predict.shape[2]))
    for k__ in range(repr_predict.shape[0]):
        if np.max(repr_predict[k__, :, :, id_max_filter[k__]]) > 0:
            max_feature_map_sample[k__] = (repr_predict[k__, :, :, id_max_filter[k__]] /
                                           np.max(repr_predict[k__, :, :, id_max_filter[k__]]))
        else:
            max_feature_map_sample[k__] = repr_predict[k__, :, :, id_max_filter[k__]]
    # normalization dependent on the filter
    print('shape', repr_predict.shape)

    for i__ in range(100):
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(test_data[i__])
        ax[1].imshow(model_.get_weights()[0][:, :, 0, id_max_filter[i__]])
        gg = ax[2].imshow(max_feature_map_sample[i__])
        fig.colorbar(gg, ax=ax[2])
        plt.tight_layout()
        plt.savefig(join(path_figure, 'figure_%i.pdf') % i__)
        plt.close()

    print(max_feature_map_sample[:, bm_1st].shape)
    print(np.max(max_feature_map_sample[:, bm_1st], axis=-1).shape)

    mean_center_1st_lst.append(np.mean(max_feature_map_sample[:, bm_1st], axis=-1))
    std_center_1st_lst.append(np.std(max_feature_map_sample[:, bm_1st], axis=-1))
    max_center_1st_lst.append(np.max(max_feature_map_sample[:, bm_1st], axis=-1))

    mean_bckgrn_1st_lst.append(np.mean(max_feature_map_sample[:, ~bm_1st], axis=-1))
    std_bckgrn_1st_lst.append(np.std(max_feature_map_sample[:, ~bm_1st], axis=-1))
    max_bckgrn_1st_lst.append(np.max(max_feature_map_sample[:, ~bm_1st], axis=-1))

    del repr_predict, max_repr_predict, max_feature_map_sample

    np.save(join(save_path_tmp, 'layer_1', 'mean_center.npy'), np.array(mean_center_1st_lst))
    np.save(join(save_path_tmp, 'layer_1', 'std_center.npy'), np.array(std_center_1st_lst))
    np.save(join(save_path_tmp, 'layer_1', 'max_center.npy'), np.array(max_center_1st_lst))

    np.save(join(save_path_tmp, 'layer_1', 'mean_background.npy'), np.array(mean_bckgrn_1st_lst))
    np.save(join(save_path_tmp, 'layer_1', 'std_background.npy'), np.array(std_bckgrn_1st_lst))
    np.save(join(save_path_tmp, 'layer_1', 'max_background.npy'), np.array(max_bckgrn_1st_lst))

    print('done')
    sys.stdout.flush()

    del model_first_layer
    del mean_center_1st_lst
    del std_center_1st_lst
    del mean_bckgrn_1st_lst
    del std_bckgrn_1st_lst
    del max_bckgrn_1st_lst
    del max_center_1st_lst

    model_second_layer = load_second_layer(model_)
    repr_predict = model_second_layer.predict(test_data.reshape(-1, 256, 256, 1))
    max_repr_predict = np.max(repr_predict, axis=(1, 2))
    # highest activation for every image and every filter
    const_each_filter = np.max(max_repr_predict, axis=0)
    # highest activation each filter
    id_max_filter = np.argmax(max_repr_predict, axis=-1)
    # for each image, which filter gives the highest activation
    max_feature_map_sample = np.zeros((repr_predict.shape[0], repr_predict.shape[1], repr_predict.shape[2]))
    for k__ in range(repr_predict.shape[0]):
        if np.max(repr_predict[k__, :, :, id_max_filter[k__]]) > 0:
            max_feature_map_sample[k__] = (repr_predict[k__, :, :, id_max_filter[k__]] /
                                           np.max(repr_predict[k__, :, :, id_max_filter[k__]]))
        else:
            max_feature_map_sample[k__] = repr_predict[k__, :, :, id_max_filter[k__]]
    # normalization dependent on the filter
    print('shape', repr_predict.shape)

    # normalization dependent on the filter

    mean_center_2nd_lst.append(np.mean(max_feature_map_sample[:, bm_2nd], axis=-1))
    std_center_2nd_lst.append(np.std(max_feature_map_sample[:, bm_2nd], axis=-1))
    max_center_2nd_lst.append(np.max(max_feature_map_sample[:, bm_2nd], axis=-1))

    mean_bckgrn_2nd_lst.append(np.mean(max_feature_map_sample[:, ~bm_2nd], axis=-1))
    std_bckgrn_2nd_lst.append(np.std(max_feature_map_sample[:, ~bm_2nd], axis=-1))
    max_bckgrn_2nd_lst.append(np.max(max_feature_map_sample[:, ~bm_2nd], axis=-1))

    del repr_predict, max_repr_predict, max_feature_map_sample

    np.save(join(save_path_tmp, 'layer_2', 'mean_center.npy'), np.array(mean_center_2nd_lst))
    np.save(join(save_path_tmp, 'layer_2', 'std_center.npy'), np.array(std_center_2nd_lst))
    np.save(join(save_path_tmp, 'layer_2', 'max_center.npy'), np.array(max_center_2nd_lst))

    np.save(join(save_path_tmp, 'layer_2', 'mean_background.npy'), np.array(mean_bckgrn_2nd_lst))
    np.save(join(save_path_tmp, 'layer_2', 'std_background.npy'), np.array(std_bckgrn_2nd_lst))
    np.save(join(save_path_tmp, 'layer_2', 'max_background.npy'), np.array(max_bckgrn_2nd_lst))

    del model_second_layer
    del mean_center_2nd_lst
    del std_center_2nd_lst
    del mean_bckgrn_2nd_lst
    del std_bckgrn_2nd_lst
    del max_bckgrn_2nd_lst
    del max_center_2nd_lst

    model_third_layer = load_third_layer(model_)
    repr_predict = model_third_layer.predict(test_data.reshape(-1, 256, 256, 1))
    max_repr_predict = np.max(repr_predict, axis=(1, 2))
    # highest activation for every image and every filter
    const_each_filter = np.max(max_repr_predict, axis=0)
    # highest activation each filter
    id_max_filter = np.argmax(max_repr_predict, axis=-1)
    # for each image, which filter gives the highest activation
    max_feature_map_sample = np.zeros((repr_predict.shape[0], repr_predict.shape[1], repr_predict.shape[2]))
    for k__ in range(repr_predict.shape[0]):
        if np.max(repr_predict[k__, :, :, id_max_filter[k__]]) > 0:
            max_feature_map_sample[k__] = (repr_predict[k__, :, :, id_max_filter[k__]] /
                                           np.max(repr_predict[k__, :, :, id_max_filter[k__]]))
        else:
            max_feature_map_sample[k__] = repr_predict[k__, :, :, id_max_filter[k__]]
    # normalization dependent on the filter

    mean_center_3rd_lst.append(np.mean(max_feature_map_sample[:, bm_3rd], axis=-1))
    std_center_3rd_lst.append(np.std(max_feature_map_sample[:, bm_3rd], axis=-1))
    max_center_3rd_lst.append(np.max(max_feature_map_sample[:, bm_3rd], axis=-1))

    mean_bckgrn_3rd_lst.append(np.mean(max_feature_map_sample[:, ~bm_3rd], axis=-1))
    std_bckgrn_3rd_lst.append(np.std(max_feature_map_sample[:, ~bm_3rd], axis=-1))
    max_bckgrn_3rd_lst.append(np.max(max_feature_map_sample[:, ~bm_3rd], axis=-1))

    del repr_predict, max_repr_predict, max_feature_map_sample

    np.save(join(save_path_tmp, 'layer_3', 'mean_center.npy'), np.array(mean_center_3rd_lst))
    np.save(join(save_path_tmp, 'layer_3', 'std_center.npy'), np.array(std_center_3rd_lst))
    np.save(join(save_path_tmp, 'layer_3', 'max_center.npy'), np.array(max_center_3rd_lst))

    np.save(join(save_path_tmp, 'layer_3', 'mean_background.npy'), np.array(mean_bckgrn_3rd_lst))
    np.save(join(save_path_tmp, 'layer_3', 'std_background.npy'), np.array(std_bckgrn_3rd_lst))
    np.save(join(save_path_tmp, 'layer_3', 'max_background.npy'), np.array(max_bckgrn_3rd_lst))

    del model_third_layer
    del mean_center_3rd_lst
    del std_center_3rd_lst
    del mean_bckgrn_3rd_lst
    del std_bckgrn_3rd_lst
    del max_bckgrn_3rd_lst
    del max_center_3rd_lst """

