import numpy

from stegano import exifHeader
from stegano import lsbset

import matplotlib.pyplot as plt
import skimage.io as io

#*******************************************************************************
def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])

#*******************************************************************************
class Comparer(object):
    """"""
    def __init__(self, filepath_cover, filepath_stego):
        super(Comparer, self).__init__()
        self._filepath_cover = filepath_cover
        self._filepath_stego = filepath_stego

        self.I = io.imread(filepath_cover)
        self.S = io.imread(filepath_stego)

    @property
    def diff(self):
        return self.I - self.S

    @ property
    def reveal(self):
        return exifHeader.reveal(self._filepath_stego)

    def print_details(self):
        print 'cover_image: %s' % self._filepath_cover
        print 'stego_image: %s' % self._filepath_stego
        print 'message length = %s' % len(self.reveal)
        a = numpy.sum(abs(self.diff))
        print 'sum of absolute image difference = %s' % a

    def plot_images(self):
        io.imshow(numpy.concatenate((self.I, self.S), axis=1))
        plt.title('Left: original cover image. Right: steganographic image.')
        plt.show()

    def plot_difference(self, absolute=False):
        io.imshow(self.diff if not absolute else abs(self.diff))
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

    # Z = Comparer(path_cover + filename, path_stego + filename)
    # Z = Comparer(
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
    # Z = Comparer(
    #     "{}images/Lenna.jpg".format(path),
    #     "{}images/Lenna_stego.jpg".format(path)
    # )


    path_cover = '{}images/png/cover/'.format(path)
    path_stego = '{}images/png/stego/'.format(path)

    filename = 'cat.445.png'

    I = io.imread(path_cover + filename)
    S = io.imread(path_stego + filename)
