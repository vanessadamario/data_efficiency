import sys
import numpy as np
from os.path import join
from PIL import Image
from tensorflow.keras.datasets.mnist import load_data


def upscale(x, output_dim):
    image = Image.fromarray(x)
    tmp_x = image.resize(size=(output_dim, output_dim))
    return np.array(tmp_x)


def main():
    new_mnist_dims = int(sys.argv[1])
    (x_train_mnist, y_train), (x_test_mnist, y_test) = load_data()

    # new_mnist_dims = 28  # 200  # 28, 80, 150, 200
    places_dims = 256

    root_path = '/raid/poggio/home/vanessad/data/mnist_natural/exp_4/dim_%i' % new_mnist_dims
    path_save_train = join(root_path, 'train')
    path_save_test = join(root_path, 'test')

    mesh_x, mesh_y = np.meshgrid(np.arange(places_dims//2 - new_mnist_dims//2,
                                           new_mnist_dims//2 + places_dims//2),
                                 np.arange(places_dims//2 - new_mnist_dims//2,
                                           new_mnist_dims//2 + places_dims//2))

    for path_, x_ in zip([path_save_test, path_save_train],
                         [x_test_mnist, x_train_mnist]):
        for k_split_ in range(100):
            split = np.load(join(path_, "split_%i.npy" % k_split_))
            n_split = split.shape[0]
            copy_split = np.copy(split)
            print("COPY SPLIT", copy_split.shape)
            copy_split[:, mesh_x, mesh_y] = np.zeros((n_split,
                                                      new_mnist_dims,
                                                      new_mnist_dims))
            max_value = np.max(copy_split, axis=(1, 2))
            print("max value shape", max_value.shape)
            print("max value", np.max(max_value))
            max_value[max_value < 1] = 1
            print("max value", np.max(max_value))

            for id_ in range(n_split):
                split[id_] = split[id_] / max_value[id_]
                if new_mnist_dims > 28:
                    x_id_ = upscale(x_[n_split * k_split_ + id_], new_mnist_dims)  # we upscale if needed
                else:
                    x_id_ = x_[n_split * k_split_ + id_]
                split[id_, mesh_y, mesh_x] = x_id_ / 255
            np.save(join(path_, 'split_%i.npy' % k_split_), split)


if __name__ == '__main__':
    main()