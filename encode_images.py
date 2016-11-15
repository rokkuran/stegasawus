import os
import numpy

from stegano import lsb
import matplotlib.pyplot as plt


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

    print 'image encoding complete.'
    return images_failed


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

    message = get_message(filepath_message)
    encode_message(path_cropped, message * 11, path_output)
    #
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
