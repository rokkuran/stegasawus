import os
import numpy

from stegano import lsb
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from skimage import io


def get_message(filepath):
    with open(filepath, 'rb') as f:
        message = f.read()
    return message

def encode_message(path_images, message, path_output):
    print 'encoding images...'
    images_failed = list()
    for i, filename in enumerate(os.listdir(path_images)):
        try:
            filepath_image = '{}{}'.format(path_images, filename)
            secret = lsb.hide(filepath_image, message, auto_convert_rgb=True)
            secret.save('{}{}'.format(path_output, filename))
            print '{}: {}'.format(i, filename)
        except IOError:
            print '{}: {} FAILED'.format(i, filename)
            images_failed.append(filename)
        except Exception:
            print '{}: {} FAILED MESSAGE TOO LONG'.format(i, filename)
            images_failed.append(filename)

    print 'image encoding complete.'
    return images_failed


def plot_embedding_locations(image1, image2, multiplier=1):
    p = numpy.concatenate((image1, image2), axis=1)
    diff = numpy.concatenate(
        ((image2 - image1) * multiplier, abs(image2 - image1) * multiplier),
        axis=1
    )
    p = numpy.concatenate((p, diff), axis=0)
    plt.imshow(p)
    plt.grid(False)
    plt.show()

def image_statistics(x, label):
    feature_functions = [
        ('mean', numpy.mean),
        ('stdev', numpy.std),
        ('skew', stats.skew),
        ('kurtosis', stats.kurtosis)]

    for fname, fn in feature_functions:
        print '{} | {} = {}'.format(label, fname, fn(x.flatten()))


def plot_rgb_embeddings(image1, image2):
    greyscale1 = rgb_to_grey(image1)
    greyscale2 = rgb_to_grey(image2)
    r1, g1, b1 = image1[:, :, 0], image1[:, :, 1], image1[:, :, 2]
    r2, g2, b2 = image2[:, :, 0], image2[:, :, 1], image2[:, :, 2]

    print 'Original Image Statistics:'
    image_statistics(greyscale1, 'greyscale1')
    image_statistics(greyscale2, 'greyscale2')
    image_statistics(r1, 'r1')
    image_statistics(r2, 'r2')
    image_statistics(g1, 'g1')
    image_statistics(g2, 'g2')
    image_statistics(b1, 'b1')
    image_statistics(b2, 'b2')

    row1 = numpy.concatenate((greyscale2 - greyscale1, r2 - r1), axis=1)
    row2 = numpy.concatenate((g2 - g1, b2 - b1), axis=1)
    p = numpy.concatenate((row1, row2), axis=0)

    plt.grid(False)
    plt.imshow(p)
    plt.show()


def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    filepath_message = '{}/message.txt'.format(path)

    # training dataset
    path_images = '{}/images/originals/'.format(path)
    path_cropped = '{}/images/train/cropped/'.format(path)
    path_output = '{}/images/train/encoded/'.format(path)

    # validation dataset
    # path_images = '/home/rokkuran/workspace/kaggle/painter_by_numbers/train_2/'
    # path_cropped = '{}/images/validation/cropped/'.format(path)
    # path_output = '{}/images/validation/encoded/'.format(path)
    #

    # path_images = '/home/rokkuran/workspace/kaggle/cats_vs_dogs/train/cats/'
    path_cover = '{}/images/train_catdog/cover/'.format(path)
    path_stego = '{}/images/train_catdog/stego/'.format(path)

    # message = get_message(filepath_message)
    # encode_message(path_cover, message * 11, path_stego)
    # #
    # path_images = '{}/images/train/cropped/'.format(path)
    # path_output = '{}/images/'.format(path)
    # # filename = '19694.jpg'
    # filename = '16.jpg'
    # # image = io.imread(fname='{}{}'.format(path_images, filename))
    # # io.imsave(path_output + filename, image)
    #
    # multipliers = [1, 11]
    # for m in multipliers:
    #     filepath_image = '{}{}'.format(path_images, filename)
    #     secret = lsb.hide(filepath_image, message * m, auto_convert_rgb=True)
    #     secret.save('{}{}_{}.jpg'.format(path_output, filename.split('.')[0], m))
    #
    #
    # image1 = io.imread(fname='{}{}_{}.jpg'.format(path_output, filename.split('.')[0], 1))
    # image2 = io.imread(fname='{}{}_{}.jpg'.format(path_output, filename.split('.')[0], 11))
    #
    # # plot_image = numpy.concatenate((image1, image2), axis=1)
    # plot_image = numpy.concatenate((image, cA, cD), axis=1)
    # plt.imshow(plot_image)
    # plt.show()

    # filename = '14.jpg'
    filename = 'cat.698.jpg'
    # image1 = io.imread('{}{}'.format(path_cropped, filename), as_grey=True)
    # image2 = io.imread('{}{}'.format(path_output, filename), as_grey=True)

    # image1 = io.imread('{}{}'.format(path_cropped, filename))
    # image2 = io.imread('{}{}'.format(path_output, filename))
    image1 = io.imread('{}{}'.format(path_cover, filename))
    image2 = io.imread('{}{}'.format(path_stego, filename))
    # image1 = rgb_to_grey(image1)
    # image2 = rgb_to_grey(image2)
    plot_embedding_locations(image1, image2, multiplier=1)
    # plot_embedding_locations(image1[:, :, 0], image2[:, :, 0], multiplier=1)
    # plot_rgb_embeddings(image1, image2)

    # x = numpy.linspace(-500, 500, 1000)
    # plt.plot(x, mlab.normpdf(x, 130.891220093, 67.3387022185), color='k')
    # plt.plot(x, mlab.normpdf(x, 130.885940552, 67.3199700626), color='r')
    # plt.show()
