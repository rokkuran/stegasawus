import numpy
import pandas

from stegano import exifHeader
from stegano import lsbset
from stegano.lsbset import generators

from scipy import stats

import matplotlib.pyplot as plt
import skimage.io as io


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
        plt.show()

    def plot_difference(self, absolute=False):
        io.imshow(self.diff if not absolute else abs(self.diff))
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


def generate_log_feature_histograms(filepath_train, bins=20):
    """"""
    train = pandas.read_csv(filepath_train)

    for feature in [x for x in train.columns if x not in ('label', 'image')]:
        try:
            plt.hist(
                train[train.label=='cover'][feature].apply(numpy.log1p),
                bins=bins, color='b', alpha=0.3, edgecolor='None'
            )

            plt.hist(
                train[train.label=='stego'][feature].apply(numpy.log1p),
                bins=bins, color='r', alpha=0.3, edgecolor='None'
            )
            # plt.ylim([0, 50])
            plt.legend(loc='upper right')
            plt.title(feature)
            plt.savefig('{}/output/{}_bins{}_log1p.png'.format(path, feature, bins))
            plt.close()
            print feature
        except ValueError as e:
            print feature, e



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


    # path_cover = '{}images/png/cover/'.format(path)
    # path_stego = '{}images/png/stego/'.format(path)
    #
    # filename = 'cat.13.png' # 96, 110, 224, 725
    #
    # z = ImageComparer(path_cover + filename, path_stego + filename)
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

    generate_feature_histograms(
        '{}data/train.csv'.format(path),
        bins=50,
        normalise=False
    )
    # generate_log_feature_histograms('{}data/train.csv'.format(path), bins=50)
