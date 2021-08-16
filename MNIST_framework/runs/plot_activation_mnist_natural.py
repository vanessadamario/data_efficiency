import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    layer_n_lst = [1, 2, 3]
    n_tr_lst = [50, 100, 300, 1000]

    dct_palette = {1: sns.light_palette(color='darkorange', n_colors=5)[2:],
                   0: sns.light_palette(color='grey', n_colors=5)[2:]}

    dct_title = {1: 'First layer',
                 2: 'Second layer',
                 3: 'Third layer'}

    flag = 'max'  # 'max' or 'mean'

    for l in layer_n_lst:
        path_lst = [
            join('/om/user/vanessad/MNIST_framework/runs/activations_mnist_natural/filters_n_tr_%i/layer_%i' % (n_tr, l))
            for n_tr in n_tr_lst]

        mean_bck_lst = []
        std_bck_lst = []
        mean_cnt_lst = []
        std_cnt_lst = []

        for path_ in path_lst:

            _bck = np.squeeze(np.load(join(path_, '%s_background.npy' % flag)))  # de-comment for std
            _cnt = np.squeeze(np.load(join(path_, '%s_center.npy' % flag)))

            # mean over the bck features and the different images
            # mean_bck_lst.append(np.mean(mean_bck.reshape(-1, )))
            # mean_cnt_lst.append(np.mean(mean_cnt.reshape(-1, )))
            # std_bck_lst.append(np.std(mean_bck.reshape(-1, )))
            # std_cnt_lst.append(np.std(mean_cnt.reshape(-1, )))

            mean_bck_lst.append(np.mean(_bck))
            mean_cnt_lst.append(np.mean(_cnt))
            std_bck_lst.append(np.std(_bck))
            std_cnt_lst.append(np.std(_cnt))

        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titleweight'] = 'bold'

        plt.rc('xtick', labelsize=40)
        plt.rc('ytick', labelsize=40)
        plt.rc('axes', labelsize=40)
        plt.rcParams['legend.title_fontsize'] = 40

        plt.figure(figsize=(15, 10))
        plt.xscale("log", nonposx='clip')
        sns.set_style(style='whitegrid')

        plt.errorbar(n_tr_lst,
                     mean_bck_lst,
                     std_bck_lst,
                     label='bkgr',
                     linewidth=20,
                     marker='o',
                     color=dct_palette[0][l-1])
        plt.errorbar(n_tr_lst,
                     mean_cnt_lst,
                     std_cnt_lst,
                     label='cntr',
                     linewidth=20,
                     marker='o',
                     color=dct_palette[1][l - 1])
        plt.xlabel('# training examples\nper class', fontsize=60)
        plt.ylabel('%s-activation' % flag, fontsize=60)
        plt.title(dct_title[l], fontsize=60, pad=100)

        print(mean_cnt_lst)
        print(std_cnt_lst)

        plt.subplots_adjust(top=0.85)
        sns.despine()
        plt.legend(bbox_to_anchor=(0.5, 1.16),
                   loc='upper center',
                   title='pixels contribution',
                   borderaxespad=0.,
                   fontsize=35,
                   fancybox=True,
                   ncol=2)
        plt.tight_layout()
        plt.savefig('max_filter_%s_position_layer_%i.pdf' % (flag, l))
        plt.close()


if __name__ == "__main__":
    main()