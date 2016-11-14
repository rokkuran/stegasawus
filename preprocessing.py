import os
import numpy

import skimage
import skimage.io as io

from skimage import transform
from scipy import stats

import matplotlib.pyplot as plt


def pyramid(image, max_layer=3, downscale=2):
    pyramid = tuple(transform.pyramid_gaussian(
        image=image,
        max_layer=max_layer,
        downscale=2
    ))
    return pyramid

def pyramid_residuals(pyramid):
    # First layer in pyramid is original image. Calculate residuals between
    # filtered downsampled images subsequently rescaled and the original.
    residuals = []
    for x in pyramid[1:]:
        x_resized = transform.resize(x, pyramid[0].shape)
        residuals.append(pyramid[0] - x_resized)
    return residuals

def plot_pyramid_residuals(arg):
    # original then 3 residuals
    pass

def get_feature_vector(a):
    # Calculate statistics
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
    for i, layer in enumerate(p):
        feature_names, feature_vector = get_feature_vector(layer)
        for metric, value in zip(feature_names, feature_vector):
            feature_name = '{}_{}_{}'.format(pyramid_type, i, metric)
            features[feature_name] = value
    return features

def gaussian_pyramid_features(image, max_layer=3, downscale=2):
    p = pyramid(image, max_layer, downscale)
    residuals = pyramid_residuals(p)

    features = get_pyramid_features(p, pyramid_type='gp')
    features.update(get_pyramid_features(residuals, pyramid_type='gp_res'))

    return features

if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    path_images = '{}/images/originals/'.format(path)
    filename = '19694.jpg'

    image = io.imread(
        fname='{}{}'.format(path_images, filename),
        as_grey=True
    )

    features = gaussian_pyramid_features(image)
