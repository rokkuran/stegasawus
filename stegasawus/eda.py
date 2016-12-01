import numpy
import pandas
import pywt

from stegano import exifHeader
from stegano import lsbset
from stegano.lsbset import generators

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as io

# sns.set_style("white")
sns.set_style("whitegrid", {'axes.grid' : False})

#*******************************************************************************
def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])

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

    # @ property
    def reveal(self, generator):
        if self._file_type == 'jpg':
            return exifHeader.reveal(self._filepath_stego)
        elif self._file_type == 'png':
            return lsbset.reveal(self._filepath_stego, generator=generator)
        else:
            raise Exception('reveal: invalid file type.')

    def print_details(self):
        print 'cover_image: %s' % self._filepath_cover
        print 'stego_image: %s' % self._filepath_stego
        # print 'message length = %s' % len(self.reveal)
        a = numpy.sum(abs(self.diff))
        print 'sum of absolute image difference = %s' % a

    def plot_images(self):
        io.imshow(numpy.concatenate((self.I, self.S), axis=1))
        plt.title('Left: original cover image. Right: steganographic image.')
        plt.grid(False)
        plt.show()

    def plot_rgb_components(self):
        """"""
        # I_rgb = numpy.concatenate([self.I[:, :, c] for c in xrange(3)], axis=1)
        # S_rgb = numpy.concatenate([self.S[:, :, c] for c in xrange(3)], axis=1)
        # IS_rgb = numpy.concatenate((I_rgb, S_rgb), axis=0)
        # io.imshow(IS_rgb)

        f, axarr = plt.subplots(nrows=2, ncols=3)

        axarr[0, 0].imshow(self.I[:, :, 0], cmap='Reds')
        axarr[0, 0].set_title('cover red')
        axarr[0, 1].imshow(self.I[:, :, 1], cmap='Greens')
        axarr[0, 1].set_title('cover green')
        axarr[0, 2].imshow(self.I[:, :, 2], cmap='Blues')
        axarr[0, 2].set_title('cover blue')
        axarr[1, 0].imshow(self.S[:, :, 0], cmap='Reds')
        axarr[1, 0].set_title('stego red')
        axarr[1, 1].imshow(self.S[:, :, 1], cmap='Greens')
        axarr[1, 1].set_title('stego green')
        axarr[1, 2].imshow(self.S[:, :, 2], cmap='Blues')
        axarr[1, 2].set_title('stego blue')
        # axarr[1, 0].imshow(self.I[:, :, 0], cmap='green')
        # axarr[1, 0].set_title('Axis [1,0]')
        # axarr[1, 1].scatter(x, y ** 2)
        # axarr[1, 1].set_title('Axis [1,1]')

        # title = '1st row: RGB components of cover image. | '
        # title += '2nd row: RGB components of stego image.'
        # plt.title(title)
        # plt.grid(False)
        plt.show()

    def plot_difference(self, absolute=False):
        io.imshow(self.diff if not absolute else abs(self.diff))
        plt.grid(False)
        plt.show()


def generate_feature_histograms(filepath_train, bins=50, normalise=False):
    """"""
    train = pandas.read_csv(filepath_train)

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
    train = pandas.read_csv(filepath_train)

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
    train = pandas.read_csv(filepath_train)

    cols = [x for x in train.columns if x not in ('label', 'image')]

    if normalise:
        train[cols] = train[cols].apply(lambda x: (x - x.mean()) / x.std())

    for feature in cols:
        I = train[train.label=='cover'][feature]
        S = train[train.label=='stego'][feature]

        I_kde = stats.gaussian_kde(I)
        S_kde = stats.gaussian_kde(S)

        x = numpy.hstack((I, S))
        xs = numpy.linspace(min(x), max(x), 250)

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
            cAcH = numpy.concatenate((coeffs[0], cH), axis=1)
            cVcD = numpy.concatenate((cV, cD), axis=1)
            plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
        else:
            plot_image = numpy.concatenate((plot_image, cH), axis=1)
            cVcD = numpy.concatenate((cV, cD), axis=1)
            plot_image = numpy.concatenate((plot_image, cVcD), axis=0)

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
    # if numpy.sum(abs(Z.diff)) > 0:
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
