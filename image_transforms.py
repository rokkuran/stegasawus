import os
import skimage
import skimage.io as io

# from skimage.filter import guassian_filter
from skimage.transform import pyramid_gaussian


import matplotlib.pyplot as plt


def crop_images(path_images, path_output, dimensions=(256, 256)):
    print 'cropping images...'
    for i, filename in enumerate(os.listdir(path_images)):
        try:
            image = io.imread('{}{}'.format(path_images, filename))
            cropped = image[0:256, 0:256]
            io.imsave(
                fname='{}{}'.format(path_output, filename),
                arr=cropped
            )
            print '{}: {}'.format(i, filename)
        except IndexError:
            print '{}: {} failed - dimensions incompatible'.format(i, filename)

    print 'all images cropped and saved.'


path = '/home/rokkuran/workspace/stegasawus'

# training dataset
# path_images = '{}/images/originals/'.format(path)
# path_cropped = '{}/images/train/cropped/'.format(path)
# path_output = '{}/images/train/encoded/'.format(path)

# validation dataset
# path_images = '/home/rokkuran/workspace/kaggle/painter_by_numbers/train_2/'
# path_cropped = '{}/images/validation/cropped/'.format(path)
# path_output = '{}/images/validation/encoded/'.format(path)


# path_images = '/home/rokkuran/workspace/kaggle/cats_vs_dogs/train/cats/'
path_images = '/home/rokkuran/workspace/kaggle/cats_vs_dogs/train/dogs/'
# path_cover = '{}/images/train_catdog/cover/'.format(path)
# path_stego = '{}/images/train_catdog/stego/'.format(path)

path_cover = '{}/images/train_catdog_rndembed/cover/'.format(path)
path_stego = '{}/images/train_catdog_rndembed/stego/'.format(path)

crop_images(path_images, path_cover)


if __name__ == '__main__':
    pass
    # path = '/home/rokkuran/workspace/stegasawus'
    # path_images = '{}/images/originals/'.format(path)
    # filename = '19694.jpg'
    #
    #
    # io.imshow('{}{}'.format(path_images, filename))
    # plt.show()
    #
    # filename = '19694.jpg'
    # x = io.imread('{}{}'.format(path_images, filename))
    # io.imshow(x)
    # plt.show()
    # cropped = x[0:256, 0:256]
    # io.imshow(cropped)
    # plt.show()
    #
    # print x.shape
    #
    # filename = '101143.jpg'
    # x = io.imread('{}{}'.format(path_images, filename))
    # print x.shape
    #
    # filename = '17699.jpg'
    # x = io.imread('{}{}'.format(path_images, filename))
    # print x.shape
    #
    # import cv2
    # import numpy
    #
    # img = cv2.imread('{}{}'.format(path_images, filename), 0)      # 1 chan, grayscale!
    # imf = numpy.float32(img)/255.0  # float conversion/scale
    # dst = cv2.dct(imf)           # the dct
    # img = numpy.uint8(dst)*255.0    # convert back
