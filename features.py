import numpy
import pandas
import os
import pywt

from scipy import stats
import matplotlib.pyplot as plt
from skimage import io

#*******************************************************************************
def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])


def statistical_metrics(values, name):
    metrics = (
        ('mean', numpy.mean),
        ('stdev', numpy.std),
        ('skew', stats.skew),
        ('kurtosis', stats.kurtosis)
    )

    features = {}
    for f, fn in metrics:
        features['{}_{}'.format(name, f)] = fn(values.flatten())

    return features

# lags=((1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 2))
def autocorrelation_features(I, lags=((1, 0), (0, 1))):
    features = {}
    m, n = I.shape

    for x, y in lags:
        ac = I[x:, y:] * I[:m-x, :n-y]
        aca = ac / (I[x:, y:].std() * I[:m-x, :n-y].std())
        features['aca_{}{}'.format(x, y)] = aca.sum()

        sm = statistical_metrics(values=ac, name='ac_{}{}'.format(x, y))
        features.update(sm)

    return features


def rgb_autocorrelation_features(I, lags=((1, 0), (0, 1))):
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        for x, y in lags:
            ac = I[x:, y:, c] * I[:m-x, :n-y, c]
            aca = ac / (I[x:, y:, c].std() * I[:m-x, :n-y, c].std())
            features['aca_{}_{}{}'.format(colour, x, y)] = aca.sum()

            sm = statistical_metrics(ac, 'ac_{}_{}{}'.format(colour, x, y))
            features.update(sm)

    return features


def create_image_ac_feature_dataset(path_images, class_label, path_output,
    image_limit=None):
    """Create autocorrelation feature set from images."""

    print 'creating image feature dataset...'

    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        fname = '{}{}'.format(path_images, filename)
        # image = io.imread(fname, as_grey=True)
        image = io.imread(fname)

        lags = ((1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 2))
        # lags = ((1, 0), (0, 1))
        # features = autocorrelation_features(image, lags=lags)
        features = rgb_autocorrelation_features(image, lags=lags)
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


def create_training_set(filepath_cover, filepath_stego, path_output):
    """"""
    train_cover = pandas.read_csv(filepath_cover)
    train_stego = pandas.read_csv(filepath_stego)
    train = pandas.concat([train_cover, train_stego])
    train.to_csv(path_output, index=False)
    return train


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


def create_image_wavelet_feature_dataset(path_images, class_label, path_output,
    image_limit=None):
    """Create autocorrelation feature set from images."""

    print 'creating image feature dataset...'

    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        fname = '{}{}'.format(path_images, filename)
        # image = io.imread(fname, as_grey=True)
        image = io.imread(fname)

        image_greyscale = rgb_to_grey(image)
        coeffs = pywt.wavedec2(image_greyscale, wavelet='haar', level=3)
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

#*******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus/'
    # path_cover = '{}images/train_catdog/cover/'.format(path)
    # path_stego = '{}images/stego/catdog/'.format(path)

    path_cover = '{}images/png/cover/'.format(path)
    path_stego = '{}images/png/stego/'.format(path)

    # create_image_ac_feature_dataset(
    #     path_images=path_cover,
    #     class_label='cover',
    #     path_output='{}data/train_cover.csv'.format(path)
    # )
    #
    # create_image_ac_feature_dataset(
    #     path_images=path_stego,
    #     class_label='stego',
    #     path_output='{}data/train_stego.csv'.format(path)
    # )

    create_image_wavelet_feature_dataset(
        path_images=path_cover,
        class_label='cover',
        path_output='{}data/train_cover_wavelet.csv'.format(path)
    )

    create_image_wavelet_feature_dataset(
        path_images=path_stego,
        class_label='stego',
        path_output='{}data/train_stego_wavelet.csv'.format(path)
    )

    create_training_set(
        '{}data/train_cover_wavelet.csv'.format(path),
        '{}data/train_stego_wavelet.csv'.format(path),
        '{}data/train_wavelet.csv'.format(path)
    )
