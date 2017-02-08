from stegasawus import lsb, seq, dataset

import numpy as np
import pandas as pd
import abc
import pywt
import base64
import cStringIO
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as io

from os import path

sns.set_style('whitegrid', {'axes.grid': False})


def rgb_to_grey(image):
    """
    Converts RGB image array (m, n, 3) to greyscale (m, n).
    """
    return np.dot(image, [0.2989, 0.5870, 0.1140])


class ImagePlots(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def I(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def S(self):
        raise NotImplementedError()

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

    def plot_rgb_difference(self):
        """
        Plots difference between cover and steganographic images for each RGB
        colour channel.
        """
        f, axarr = plt.subplots(1, 3, figsize=(12, 4))
        for j, colour in enumerate(['Red', 'Green', 'Blue']):
            diff = self.I[:, :, j] - self.S[:, :, j]
            axarr[j].imshow(diff, cmap='{}s_r'.format(colour))
            axarr[j].set_title('{}'.format(colour))
            axarr[j].set_xticklabels([])
            axarr[j].set_yticklabels([])
        plt.show()

    def plot_difference(self):
        """
        Plot difference between cover and steganographic image.
        """
        io.imshow(self.I - self.S)
        plt.grid(False)
        plt.show()


def plot_wavelet_decomposition(image, level=3):
    """
    Plot of 2D wavelet decompositions for given number of levels.

    image needs to be either a colour channel or greyscale image:
        rgb: self.I[:, :, n], where n = {0, 1, 2}
        greyscale: use rgb_to_grey(self.I)

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
    io.imshow(abs(plot_image), cmap='gray_r')
    plt.show()


class JointImageAnalyser(ImagePlots):
    """"""
    def __init__(self, cover, stego):
        super(JointImageAnalyser, self).__init__()
        self._I = self._set_image(cover)
        self._S = self._set_image(stego)

    def _check_type(self, v, types):
        return any([isinstance(v, t) for t in types])

    def _set_image(self, image):
        if self._check_type(image, [np.ndarray, list]):
            return image
        elif self._check_type(image, [str]):
            return io.imread(image)
        else:
            raise Exception('Input image type not array like or filepath.')

    @property
    def I(self):
        return self._I

    @property
    def S(self):
        return self._S

    @property
    def diff(self):
        return self.I - self.S

    def print_details(self):
        # TODO: add some more details...
        a = np.sum(abs(self.diff))
        print 'sum of absolute image difference = %s' % a

    def reveal(self, seq_method):
        return lsb.reveal(self.S, seq_method)

    def reveal_image(self, seq_method, show=False):
        s = base64.b64decode(self.reveal(seq_method))
        s = cStringIO.StringIO(s)
        I = np.array(Image.open(s))

        if show:
            io.imshow(I)
            plt.show()

        return I


if __name__ == '__main__':
    cdir = '/home/rokkuran/workspace/stegasawus/'

    fp = path.join(cdir, 'data/messages/Lenna_64x64.png')
    msg = dataset.image_to_string(fp)

    path_images = '{}images/png/cover_test/'.format(cdir)
    path_output = '{}images/png/lsb_test/'.format(cdir)

    seq_method = seq.rand_darts(seed=77)
    g = dataset.DatasetGenerator(path_images, path_output, seq_method)
    g.batch_hide_message(msg)

    filename = 'cat.117.png'
    a = JointImageAnalyser(path_images + filename, path_output + filename)
    H = a.reveal_image(seq_method)
    io.imshow(H)
    plt.show()
