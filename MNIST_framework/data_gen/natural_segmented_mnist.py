import sys
import os
import skimage
import numpy as np
import pickle
from os.path import join
from PIL import Image
from numpy.random import choice
from tensorflow.keras.datasets.mnist import load_data


def upscale(x, output_dim):
    image = Image.fromarray(x)
    tmp_x = image.resize(size=(output_dim, output_dim))
    return np.array(tmp_x)


def main():

    start_ = sys.argv[1]

    bool_test = True  # test

    factor = 100 if bool_test else 600
    start_ = int(start_) * factor
    end_ = start_ + factor

    print(type(start_))
    print(type(end_))
    print(type(bool_test))

    save_id_path = '/om5/user/vanessad/mnist_natural_segmented/exp_4'

    print('start', start_)
    print('end', end_)
    print('bool test', bool_test)

    path_places = '/om5/user/vanessad/data/vision/torralba/deeplearning/images256'
    new_mnist_dims = 200  # 28, 80, 150, 200
    places_dims = 256
    (x_train_mnist, y_train), (x_test_mnist, y_test) = load_data()
    size_split_tr = 600
    size_split_ts = 100

    id_places_train_filename = 'id_train.npy'
    id_places_test_filename = 'id_test.npy'

    path_save_train = join(save_id_path, 'dim_%i/train' % new_mnist_dims)
    path_save_test = join(save_id_path, 'dim_%i/test' % new_mnist_dims)

    lst_fls = os.listdir(save_id_path)
    if id_places_train_filename in lst_fls and id_places_test_filename in lst_fls:
        print('Im loading')
        id_place_train = np.load(join(save_id_path, id_places_train_filename))
        id_place_test = np.load(join(save_id_path, id_places_test_filename))
        lst_path = pickle.load(open(join(save_id_path, 'lst_places_path.pkl'), 'rb'))

    else:
        lst_path = []
        for letter_ in os.listdir(path_places):
            print(letter_)
            for place_ in os.listdir(join(path_places, letter_)):
                for pic_ in os.listdir(join(path_places, letter_, place_)):
                    lst_path.append(join(path_places, letter_, place_, pic_))

        print('end of listing')
        len_tr = x_train_mnist.shape[0]
        len_ts = x_test_mnist.shape[0]

        id_place_train = choice(np.arange(len(lst_path)), size=len_tr, replace=False)  # uniformly sampled
        id_place_test = choice(np.setdiff1d(np.arange(len(lst_path)), id_place_train), size=len_ts, replace=False)
        np.save(join(save_id_path, 'id_train.npy'), id_place_train)
        np.save(join(save_id_path, 'id_test.npy'), id_place_test)
        pickle.dump(lst_path, open(join(save_id_path, 'lst_places_path.pkl'), 'wb'))

    mesh_x, mesh_y = np.meshgrid(np.arange(places_dims//2 - new_mnist_dims//2, new_mnist_dims//2 + places_dims//2),
                                 np.arange(places_dims//2 - new_mnist_dims//2, new_mnist_dims//2 + places_dims//2))

    if not bool_test:
        print("training dataset")
        for id_tr_, (mnist_sample_tr, id_place_) in enumerate(zip(x_train_mnist[start_:end_],
                                                                  id_place_train[start_:end_])):

            if (id_tr_+start_) % size_split_tr == 0:  # start a new split
                print('new split', id_tr_)
                data_split = np.zeros((size_split_tr, places_dims, places_dims))
            mnist_sample_tr = mnist_sample_tr / 255  # normalize mnist [0,1]
            path_sample_place = lst_path[id_place_]  # path to image
            if path_sample_place.startswith('/om2'):
                path_sample_place = '/om5' + path_sample_place[4:]
            try:
                image = skimage.color.rgb2grey(skimage.io.imread(path_sample_place))
            except:
                print('TRAIN', path_sample_place, id_tr_+start_)
                continue
            if new_mnist_dims > 28:
                mnist_sample_tr = upscale(mnist_sample_tr, new_mnist_dims)  # we upscale if needed
            image_ = image.copy()
            bm_places = np.zeros((places_dims, places_dims), dtype=bool)  # dimensions
            print(image_.shape)
            print(mnist_sample_tr.shape)
            bm_mnist = mnist_sample_tr > 0
            bm_places[mesh_y, mesh_x] = bm_mnist
            image_[bm_places] = mnist_sample_tr[bm_mnist]
            data_split[int(id_tr_ % size_split_tr)] = image_
            if (id_tr_+start_) % size_split_tr == size_split_tr-1:
                np.save(join(path_save_train, 'split_%i.npy' % ((id_tr_+start_)//size_split_tr)),
                        data_split)
        # np.save(join(path_save_train, 'y_train.npy'), y_train)

    else:
        print("test dataset")
        for id_ts_, (mnist_sample_ts, id_place_) in enumerate(zip(x_test_mnist[start_:end_],
                                                                  id_place_test[start_:end_])):
            print(id_ts_)
            if (id_ts_+start_) % size_split_ts == 0:
                data_split = np.zeros((size_split_ts, places_dims, places_dims))
            mnist_sample_ts = mnist_sample_ts / 255
            path_sample_place = lst_path[id_place_]
            if path_sample_place.startswith('/om2'):
                path_sample_place = '/om5' + path_sample_place[4:]

            try:
                image = skimage.color.rgb2grey(skimage.io.imread(path_sample_place))
            except:
                print('TEST', path_sample_place, id_ts_+start_)
                continue
            if new_mnist_dims > 28:
                mnist_sample_ts = upscale(mnist_sample_ts, new_mnist_dims)
            image_ = image.copy()
            bm_places = np.zeros((places_dims, places_dims), dtype=bool)  # dimensions
            bm_mnist = mnist_sample_ts > 0
            bm_places[mesh_y, mesh_x] = bm_mnist
            image_[bm_places] = mnist_sample_ts[bm_mnist]

            data_split[int(id_ts_ % size_split_ts)] = image_
            if (id_ts_+start_) % size_split_ts == size_split_ts - 1:
                np.save(join(path_save_test, 'split_%i.npy' % ((id_ts_+start_) // size_split_ts)),
                        data_split)
        # np.save(join(path_save_test, 'y_test.npy'), y_test)


if __name__ == '__main__':
    main()