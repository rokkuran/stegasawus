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

sns.set_style('whitegrid', {'axes.grid': False})


# ******************************************************************************
def rgb_to_grey(image):
    """
    Converts RGB image array to greyscale.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image array with dimensions (m, n, 3).

    Returns
    -------
    Greyscale image of dimensions (m, n)

    """
    return np.dot(image, [0.2989, 0.5870, 0.1140])


class JointImageAnalyser(object):
    """
    Image comparison class for use with original 'cover' image (I) and the
    steganographic image (S).

    Parameters
    ----------
    filepath_cover : string filepath
        Filepath for cover image.
    filepath_stego : string filepath
        Filepath for steganographic image.

    Attributes
    ----------
    I : numpy.ndarray
        RGB array for cover image.
    S : numpy.ndarray
        RGB array for steganographic image.
    diff : numpy.ndarray
        Difference between cover and steganographic image RGB arrays: (I - S).

    """
    def __init__(self, filepath_cover, filepath_stego):
        super(JointImageAnalyser, self).__init__()
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

    def print_details(self):
        print 'cover_image: %s' % self._filepath_cover
        print 'stego_image: %s' % self._filepath_stego
        a = np.sum(abs(self.diff))
        print 'sum of absolute image difference = %s' % a

    def reveal(self, generator):
        """
        Reveal hidden message in steganographic image using the generator
        specified.

        Parameters
        ----------
        generator : function
            Embedding location generator function from custom functon or
            stegano.lsbset.generators. Message will not be revealed unless the
            correct generator is used.

        Returns
        -------
        Secret message :o

        """
        if self._file_type == 'jpg':
            return exifHeader.reveal(self._filepath_stego)
        elif self._file_type == 'png':
            return lsbset.reveal(self._filepath_stego, generator=generator)
        else:
            raise Exception('reveal: invalid file type.')

    def plot_images(self):
        """
        Plot cover and steganographic RGB images side by side.
        """
        io.imshow(np.concatenate((self.I, self.S), axis=1))
        plt.title('Left: original cover image. Right: steganographic image.')
        plt.grid(False)
        plt.show()

    def plot_rgb_components(self):
        """
        Plot RGB colour channels for both cover and steganographic images.
        """
        f, axarr = plt.subplots(nrows=2, ncols=3)
        for i, image_type in enumerate(['Cover', 'Stego']):
            for j, colour in enumerate(['Red', 'Green', 'Blue']):
                axarr[i, j].imshow(self.I[:, :, j], cmap='{}s'.format(colour))
                axarr[i, j].set_title('{} {}'.format(image_type, colour))
                axarr[i, j].set_xticklabels([])
                axarr[i, j].set_yticklabels([])
        plt.show()

    def plot_difference(self, absolute=False):
        """
        Plot difference between cover and steganographic image.
        """
        io.imshow(self.diff if not absolute else abs(self.diff))
        plt.grid(False)
        plt.show()


def generate_feature_histograms(filepath_train, bins=50, normalise=False):
    """
    Generate batch of comparison histograms of cover and steganographic image
    features.

    Parameters
    ----------
    filepath_train : string
        Filepath for training csv file with image features.
    bins : int, default : 50
        Number of bins for histograms.
    normalise : bool, default : False
        Apply normalisation to image features.

    Returns
    -------
    Set of comparitive histograms.

    """
    train = pd.read_csv(filepath_train)

    cols = [x for x in train.columns if x not in ('label', 'image')]

    if normalise:
        train[cols] = train[cols].apply(lambda x: (x - x.mean()) / x.std())

    for feature in cols:
        label = 'cover'
        I = train[train.label == label][feature]
        plt.hist(
            I, bins=bins, color='b', alpha=0.3, edgecolor='None', label=label
        )

        label = 'stego'
        S = train[train.label == 'stego'][feature]
        plt.hist(
            S, bins=bins, color='r', alpha=0.3, edgecolor='None', label=label
        )

        plt.legend(loc='upper right', frameon=False)
        plt.title(feature)

        # TODO: change figure output path to argument.
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
        sns.distplot(train[train.label == label][feature], label=label)

        label = 'stego'
        sns.distplot(train[train.label == label][feature], label=label)

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
        I = train[train.label == 'cover'][feature]
        S = train[train.label == 'stego'][feature]

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
    sns.jointplot(
        x, y, kind="hex", stat_func=stats.kendalltau, color="#4CB391"
    )
    plt.show()


def plot_wavelet_decomposition(image, level=3):
    """
    Plot of 2D wavelet decompositions for given number of levels.

    Parameters
    ----------
    image : numpy.ndarray
        Single channel image with dimensions (m, n).
    level : int, default : 3
        Decomposition level.

    Returns
    -------
    Shows plot of 2D wavelet decomposition.

    """
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


# ******************************************************************************
if __name__ == '__main__':
    pass
