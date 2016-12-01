import numpy as np
import pandas as pd
import pywt

from stegano import exifHeader
from stegano import lsbset
from stegano.lsbset import generators

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as io

sns.set_style('whitegrid', {'axes.grid' : False})

#*******************************************************************************
def rgb_to_grey(image):
    return np.dot(image, [0.2989, 0.5870, 0.1140])

#*******************************************************************************
class ImageComparer(object):
    """"""
    def __init__(self, filepath_cover, filepath_stego):
        super(ImageComparer, self).__init__()
        self._filepath_cover = filepath_cover
        self._filepath_stego = filepath_stego

        # check if filetypes match
        cover_ext = filepath_cover.split('.')[-1]
        stego_ext = filepath_stego.split('.')[-1]
        if cover_ext == stego_ext:
            self._file_type = filepath_cover.split('.')[-1]
        else:
            raise Exception('Error: file types are not the same.')

        self.I = io.imread(filepath_cover)
        self.S = io.imread(filepath_stego)

    @property
    def diff(self):
        return self.I - self.S

    @property
    def print_details(self):
        print 'cover_image: %s' % self._filepath_cover
        print 'stego_image: %s' % self._filepath_stego
        a = np.sum(abs(self.diff))
        print 'sum of absolute image difference = %s' % a

    def reveal(self, generator):
        if self._file_type == 'jpg':
            return exifHeader.reveal(self._filepath_stego)
        elif self._file_type == 'png':
            return lsbset.reveal(self._filepath_stego, generator=generator)
        else:
            raise Exception('reveal: invalid file type.')


    def plot_images(self):
        io.imshow(np.concatenate((self.I, self.S), axis=1))
        plt.title('Left: original cover image. Right: steganographic image.')
        plt.grid(False)
        plt.show()

    def plot_rgb_components(self):
        """"""
        f, axarr = plt.subplots(nrows=2, ncols=3)
        for i, image_type in enumerate(['Cover', 'Stego']):
            for j, colour in enumerate(['Red', 'Green', 'Blue']):
                axarr[i, j].imshow(self.I[:, :, j], cmap='{}s'.format(colour))
                axarr[i, j].set_title('{} {}'.format(image_type, colour))
                axarr[i, j].set_xticklabels([])
                axarr[i, j].set_yticklabels([])
        plt.show()

    def plot_difference(self, absolute=False):
        io.imshow(self.diff if not absolute else abs(self.diff))
        plt.grid(False)
        plt.show()


def generate_feature_histograms(filepath_train, bins=50, normalise=False):
    """"""
    train = pd.read_csv(filepath_train)

    cols = [x for x in train.columns if x not in ('label', 'image')]

    if normalise:
        train[cols] = train[cols].apply(lambda x: (x - x.mean()) / x.std())

    for feature in cols:
        label = 'cover'
        I = train[train.label==label][feature]
        plt.hist(I, bins=bins, color='b', alpha=0.3, edgecolor='None', label=label)

        label = 'stego'
        S = train[train.label=='stego'][feature]
        plt.hist(S, bins=bins, color='r', alpha=0.3, edgecolor='None', label=label)

        plt.legend(loc='upper right', frameon=False)
        plt.title(feature)
        plt.savefig('{}/output/{}_bins{}.png'.format(path, feature, bins))
        print feature
        plt.close()


def generate_feature_distplots(filepath_train, normalise=False):
    """"""
    train = pd.read_csv(filepath_train)

    cols = [x for x in train.columns if x not in ('label', 'image')]

    if normalise:
        train[cols] = train[cols].apply(lambda x: (x - x.mean()) / x.std())

    for feature in cols:
        label = 'cover'
        sns.distplot(train[train.label==label][feature], label=label)

        label = 'stego'
        sns.distplot(train[train.label==label][feature], label=label)

        plt.legend(loc='upper right', frameon=False)
        plt.title(feature)
        plt.savefig('{}/output/distplot_{}.png'.format(path, feature))
        print feature
        plt.close()


def generate_feature_kde(filepath_train, normalise=False):
    """"""
    train = pd.read_csv(filepath_train)

    cols = [x for x in train.columns if x not in ('label', 'image')]

    if normalise:
        train[cols] = train[cols].apply(lambda x: (x - x.mean()) / x.std())

    for feature in cols:
        I = train[train.label=='cover'][feature]
        S = train[train.label=='stego'][feature]

        I_kde = stats.gaussian_kde(I)
        S_kde = stats.gaussian_kde(S)

        x = np.hstack((I, S))
        xs = np.linspace(min(x), max(x), 250)

        plt.plot(xs, I_kde(xs), color='b', lw=1.5, alpha=0.3, label='cover')
        plt.plot(xs, S_kde(xs), color='r', lw=1.5, alpha=0.3, label='stego')

        plt.legend(loc='upper right', frameon=False)
        plt.title('KDE (Gaussian): {}'.format(feature))
        plt.savefig('{}/output/kde_{}.png'.format(path, feature))
        print feature
        plt.close()


def plot_joint_dist_hex(x, y):
    sns.set(style="ticks")
    sns.jointplot(x, y, kind="hex", stat_func=stats.kendalltau, color="#4CB391")
    plt.show()


def plot_wavelet_decomposition(image, level=3):
    coeffs = pywt.wavedec2(image, wavelet='haar', level=level)
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        if i == 0:
            cAcH = np.concatenate((coeffs[0], cH), axis=1)
            cVcD = np.concatenate((cV, cD), axis=1)
            plot_image = np.concatenate((cAcH, cVcD), axis=0)
        else:
            plot_image = np.concatenate((plot_image, cH), axis=1)
            cVcD = np.concatenate((cV, cD), axis=1)
            plot_image = np.concatenate((plot_image, cVcD), axis=0)

    plt.grid(False)
    # io.imshow(abs(plot_image), cmap='gray')
    io.imshow(abs(plot_image), cmap='gray_r')
    # io.imshow(plot_image)
    plt.show()


#*******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus/'
    # path_cover = '{}images/train_catdog/cover/'.format(path)
    # path_stego = '{}images/stego/catdog/'.format(path)

    # path_cover = '{}images/train/cropped/'.format(path)
    # path_stego = '{}images/stego/paintings/'.format(path)

    # filename = 'cat.698.jpg'
    # filename = 'cat.445.jpg'
    # filename = '10.jpg'
    # plot_images(filename, path_cover, path_stego)

    # Z = ImageComparer(path_cover + filename, path_stego + filename)
    # Z = ImageComparer(
    #     '/home/rokkuran/workspace/stegasawus/images/Lenna.jpg',
    #     '/home/rokkuran/workspace/stegasawus/images/image.jpg'
    # )

    # Z.print_details()
    # if np.sum(abs(Z.diff)) > 0:
    #     Z.plot_images()
    #     Z.plot_difference()

    # exifHeader.hide(
    #     "{}images/Lenna.jpg".format(path),
    #     "{}images/Lenna_stego.jpg".format(path),
    #     secret_message="Hello world!"
    # )
    #
    # Z = ImageComparer(
    #     "{}images/Lenna.jpg".format(path),
    #     "{}images/Lenna_stego.jpg".format(path)
    # )


    path_cover = '{}images/png/cover/'.format(path)
    path_stego = '{}images/png/stego/'.format(path)

    filename = 'cat.2.png' # 96, 110, 224, 725

    z = ImageComparer(path_cover + filename, path_stego + filename)
    z.plot_rgb_components()
    # plot_wavelet_decomposition(z.I[:, :, 0])
    # z.print_details()
    # # print z.reveal(generators.eratosthenes())
    # print z.reveal(generators.identity())
    # # print z.reveal(generators.Dead_Man_Walking())
    # # print z.reveal(generators.syracuse())
    #
    # z.plot_images()
    # z.plot_difference()


    # I = io.imread(path_cover + filename)
    # S = io.imread(path_stego + filename)

    # generate_feature_distplots(
    # # generate_feature_histograms(
    #     # '{}data/train.csv'.format(path),
    #     '{}data/train_wavelet.csv'.format(path),
    #     # bins=50,
    #     normalise=False
    # )
    # generate_log_feature_histograms('{}data/train.csv'.format(path), bins=50)
    # generate_feature_kde('{}data/train.csv'.format(path))
