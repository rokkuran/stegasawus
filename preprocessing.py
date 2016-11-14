import os
import numpy
import pandas

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
        ('kurtosis', stats.kurtosis),
        ('entropy', stats.entropy)
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

def create_image_feature_dataset(path_images, class_label, path_output):
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

        if i > 1000:
            break

    df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index)
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


    path_cropped = '{}/images/validation/cropped/'.format(path)
    create_image_feature_dataset(
        path_images=path_cropped,
        class_label='clean',
        path_output='{}/data/cropped.csv'.format(path)
    )
