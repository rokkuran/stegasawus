import os
import numpy
import pandas
import pywt

import skimage
import skimage.io as io

from skimage import transform
from scipy import stats

import matplotlib.pyplot as plt


def calc_pyramid(image, max_layer=3, downscale=2):
    pyramid = tuple(transform.pyramid_gaussian(
        image=image,
        max_layer=max_layer,
        downscale=2
    ))
    return pyramid

def calc_pyramid_residuals(pyramid):
    """
    First layer in pyramid is original image. Calculate residuals between
    filtered downsampled images subsequently rescaled and the original.
    """
    residuals = []
    for x in pyramid[1:]:
        x_resized = transform.resize(x, pyramid[0].shape)
        residuals.append(pyramid[0] - x_resized)
    return residuals

def plot_pyramid_residuals(arg):
    # original then 3 residuals
    pass

def get_feature_vector(a):
    feature_functions = [
        ('mean', numpy.mean),
        ('stdev', numpy.std),
        ('skew', stats.skew),
        ('kurtosis', stats.kurtosis)
        # ('entropy', stats.entropy)
    ]
    feature_names = zip(*feature_functions)[0]

    feature_vector = []
    for (feature, fn) in feature_functions:
        feature_vector.append(fn(a.flatten()))

    return feature_names, feature_vector

def get_pyramid_features(pyramid, pyramid_type):
    features = {}
    for i, layer in enumerate(pyramid):
        feature_names, feature_vector = get_feature_vector(layer)
        for metric, value in zip(feature_names, feature_vector):
            feature_name = '{}_{}_{}'.format(pyramid_type, i, metric)
            features[feature_name] = value
    return features

def gaussian_pyramid_features(image, max_layer=3, downscale=2):
    pyramid = calc_pyramid(image, max_layer, downscale)
    residuals = calc_pyramid_residuals(pyramid)

    features = get_pyramid_features(pyramid, pyramid_type='gp')
    features.update(get_pyramid_features(residuals, pyramid_type='gp_res'))

    return features

def get_image_features(image):
    features = gaussian_pyramid_features(image)
    # features.update(other_features_here)
    return features

def create_image_feature_dataset(path_images, class_label, path_output, image_limit=None):
    print 'creating image feature dataset...'
    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        image = io.imread(
            fname='{}{}'.format(path_images, filename),
            as_grey=True
        )
        features = get_image_features(image)
        if i == 0:
            feature_names = features.keys()

        row = [filename, class_label]
        for feature in feature_names:
            row.append(features[feature])

        dataset.append(row)

        if i % 250 == 0:
            print '{} images processed'.format(i)

        if image_limit:
            if i > image_limit:
                break

    df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index=False)
    print 'image feature dataset created.'


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'

    def test_feature_generation():
        path_images = '{}/images/originals/'.format(path)
        filename = '19694.jpg'

        image = io.imread(
            fname='{}{}'.format(path_images, filename),
            as_grey=True
        )
        features = gaussian_pyramid_features(image)
        return features

    # path_cropped = '{}/images/train/cropped/'.format(path)
    # create_image_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='clean',
    #     path_output='{}/data/train_cropped.csv'.format(path)
    # )
    #
    # path_encoded = '{}/images/train/encoded/'.format(path)
    # create_image_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='message',
    #     path_output='{}/data/train_encoded.csv'.format(path)
    # )

    def create_training_set(path_output):
        train_clean = pandas.read_csv('{}/data/train_cropped.csv'.format(path))
        train_encoded = pandas.read_csv('{}/data/train_encoded.csv'.format(path))
        train = pandas.concat([train_clean, train_encoded])
        train.to_csv(path_output, index=False)
        return train

    # create_training_set('{}/data/train.csv'.format(path))

    def plot_dwt(image):
        cA, cD = pywt.dwt(image, 'haar')
        print 'shape | image {}; cA {}; cD {}'.format(image.shape, cA.shape, cD.shape)
        plot_image = numpy.concatenate((cA, cD), axis=1)
        plt.imshow(plot_image)
        plt.show()

    def plot_dwt2(image):
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        print 'shape | image {}; cA {}; cD {}'.format(image.shape, cA.shape, cD.shape)
        cAcH = numpy.concatenate((abs(cA), abs(cH)), axis=1)
        cVcD = numpy.concatenate((abs(cV), abs(cD)), axis=1)
        plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
        plt.imshow(plot_image)
        plt.show()

    def plot_dwt_coefficients(coeffs):
        cA, (cH, cV, cD) = coeffs
        print 'shape | cA {}; cH {}; cV {}; cD {}'.format(
            cA.shape, cH.shape, cV.shape, cD.shape
        )
        cAcH = numpy.concatenate((cA, cH), axis=1)
        cVcD = numpy.concatenate((cV, cD), axis=1)
        plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
        plt.imshow(plot_image)
        plt.show()

    def dwt_levels(image):
        coeffs = pywt.wavedec2(image, wavelet='haar', level=3)
        return coeffs

    # path_output = '{}/images/'.format(path)
    # filename = '18_1.jpg'
    # # filename = '17_1.jpg'
    # image = io.imread(fname='{}{}'.format(path_output, filename), as_grey=True)
    # io.imshow(image)
    # plt.show()
    # # plot_dwt(image)
    # # plot_dwt2(image)

    def rgb_to_grey(image):
        return numpy.dot(image, [0.2989, 0.5870, 0.1140])

    def plot_wavelet_decomposition(image, coeffs):
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
        io.imshow(plot_image)#, cmap='gray')
        plt.show()

    path_output = '{}/images/'.format(path)
    filename = '17_11.jpg'
    # image = io.imread(fname='{}{}'.format(path_output, filename), as_grey=True)
    image = io.imread(fname='{}{}'.format(path_output, filename))[:, :, 0]
    coeffs = pywt.wavedec2(image, wavelet='haar', level=3)
    plot_wavelet_decomposition(image, coeffs)

    def create_feature_name(layer, c, fname):
        return 'dwt_{layer}_{c}_{fname}'.format(layer=layer, c=c, fname=fname)


    def get_wavdec_feature_vector(coeffs):
        feature_functions = [
            ('mean', numpy.mean),
            ('stdev', numpy.std),
            ('skew', stats.skew),
            ('kurtosis', stats.kurtosis)]

        feature_vector = {}
        cA = coeffs[0]
        for (fname, fn) in feature_functions:
            feature_name = 'dwt_{layer}_cA_{fname}'.format(
                layer=len(coeffs) - 1, fname=fname
            )
            # reduce sensitivity to noise
            c_tol = abs(cA) > 1 # coefficients with magnitude > 1 allowed
            if c_tol.any():
                feature_vector[feature_name] = fn(cA[c_tol].flatten())
            else:
                feature_vector[feature_name] = 0

        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            layer = len(coeffs) - 1 - i

            for (fname, fn) in feature_functions:
                for c, cX in zip(('cH', 'cV', 'cD'), (cH, cV, cD)):
                    feature_name = 'dwt_{layer}_{c}_{fname}'.format(
                        layer=layer, c=c, fname=fname
                    )
                    c_tol = abs(cX) > 1
                    if c_tol.any():
                        feature_vector[feature_name] = fn(cX[c_tol].flatten())
                    else:
                        feature_vector[feature_name] = 0

        return feature_vector

    # feature_vector = get_wavdec_feature_vector(coeffs)


    def create_image_wavdec_feature_dataset(path_images, class_label, path_output, image_limit=None):
        print 'creating image feature dataset...'
        dataset = list()
        for i, filename in enumerate(os.listdir(path_images)):
            image = io.imread(
                fname='{}{}'.format(path_images, filename)
                # as_grey=True
            )
            if len(image.shape) == 3:   # make sure rgb image arrays
                coeffs = pywt.wavedec2(image[:, :, 0], wavelet='haar', level=3)
                features = get_wavdec_feature_vector(coeffs)
                if i == 0:
                    feature_names = features.keys()

                row = [filename, class_label]
                for feature in feature_names:
                    row.append(features[feature])

                dataset.append(row)

            if i % 250 == 0:
                print '{} images processed'.format(i)

            if image_limit:
                if i > image_limit:
                    break

        df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
        df.to_csv(path_output, index=False)
        print 'image feature dataset created.'


    # path_cropped = '{}/images/train/cropped/'.format(path)
    # create_image_wavdec_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='clean',
    #     path_output='{}/data/train_cropped.csv'.format(path)
    # )
    #
    # path_encoded = '{}/images/train/encoded/'.format(path)
    # create_image_wavdec_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='message',
    #     path_output='{}/data/train_encoded.csv'.format(path)
    # )
    #
    # create_training_set('{}/data/train.csv'.format(path))
