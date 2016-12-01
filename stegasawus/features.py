import numpy as np
import pandas as pd
import os
import pywt

from scipy import stats
import matplotlib.pyplot as plt
from skimage import io

#*******************************************************************************
def rgb_to_grey(image):
    """
    Converts RGB image to greyscale.

    Parameters
    ----------
    image : array
        RGB array

    Returns
    -------
    numpy.ndarray
        Greyscale image array from RGB

    """
    return np.dot(image, [0.2989, 0.5870, 0.1140])


def statistical_metrics(x):
    """
    Calculates statistical metrics on input array (mean, std, skew, kurtosis).

    Parameters
    ----------
    x : numpy.ndarray
        Array to compute statics on.

    Returns
    -------
    features : dict
        Dictionary of metrics.

    """

    metrics = {
        'mean': np.mean,
        'stdev': np.std,
        'skew': stats.skew,
        'kurtosis': stats.kurtosis
    }
    return {k: fn(x.flatten()) for k, fn in metrics.items()}


def prefix_dict_keys(d, prefix):
    return {'{}_{}'.format(prefix, k): v for k, v in d.items()}


def autocorrelation_features(I, lags=((1, 0), (0, 1))):
    """
    Calculate the autocorrelation statistical features (specified in
    statistical_metrics function) from a 2D image array for the specified lags.

    Parameters
    ----------
    I : 2D array
        Array from a greyscale image or an individual colour channel.
    lags : array of coordinate shift items
        Defaults to 1 pixel vertical and horizontal lags - ((1, 0), (0, 1)).

    Returns
    -------
    features : dict

    """

    m, n = I.shape

    features = {}
    for x, y in lags:
        ac = I[x:, y:] * I[:m-x, :n-y]
        aca = np.sum(ac / (I[x:, y:].std() * I[:m-x, :n-y].std()))

        features['aca_{}{}'.format(x, y)] = aca

        f_stat = statistical_metrics(ac)
        f_stat = prefix_dict_keys(f_stat, 'ac_{}{}'.format(x, y))
        features.update(f_stat)

    return features


def rgb_autocorrelation_features(I, lags=((1, 0), (0, 1))):
    """
    Calculate the autocorrelation statistical features of a RGB image
    array (m, n, 3) for the specified lags.

    Parameters
    ----------
    I : array
        RGB image array (m, n, 3).
    lags : array of coordinate shift items
        Defaults to 1 pixel vertical and horizontal lags - ((1, 0), (0, 1)).

    Returns
    -------
    features : dict

    """
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        f_ac = autocorrelation_features(I[:, :, c], lags)
        f_ac = prefix_dict_keys(f_ac, colour)
        features.update(f_ac)

    return features


def autocorrelation_feature_dataset(path_images, class_label, path_output,
    image_limit=None):
    """Create autocorrelation feature set from images."""

    print 'creating image feature dataset...'

    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        fname = '{}{}'.format(path_images, filename)
        image = io.imread(fname)

        lags = ((1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 2))
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

    df = pd.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index=False)

    print 'image feature dataset created.'


def create_training_set(filepath_cover, filepath_stego, path_output):
    """"""
    train_cover = pd.read_csv(filepath_cover)
    train_stego = pd.read_csv(filepath_stego)
    train = pd.concat([train_cover, train_stego])
    train.to_csv(path_output, index=False)
    return train


def apply_tolerance(x, tol):
    x_tol = abs(x) > tol
    if x_tol.any():
        return x[x_tol]
    else:
        return np.zeros(1)


def wavdec_features(coeffs, tol=1):
    """"""
    n_layers = len(coeffs) - 1

    features = {}

    cA = coeffs[0]
    prefix = 'dwt_{}_cA'.format(n_layers)
    cA = apply_tolerance(cA, tol) # reduce sensitivity to noise
    f_stat = statistical_metrics(cA)
    f_stat = prefix_dict_keys(f_stat, prefix)
    features.update(f_stat)

    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        layer = n_layers - i
        for c, cX in zip(('cH', 'cV', 'cD'), (cH, cV, cD)):
            prefix = 'dwt_{}_{}'.format(layer, c)
            cX = apply_tolerance(cX, tol)
            f_stat = statistical_metrics(cX)
            f_stat = prefix_dict_keys(f_stat, prefix)
            features.update(f_stat)

    return features


def rgb_wavelet_features(I, tol=1):
    """"""
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        coeffs = pywt.wavedec2(I[:, :, c], wavelet='haar', level=3)
        f_wavelet = wavdec_features(coeffs)
        f_wavelet = prefix_dict_keys(f_wavelet, colour)
        features.update(f_wavelet)

    return features


def wavelet_feature_dataset(path_images, class_label, path_output,
    image_limit=None):

    print 'creating image feature dataset...'

    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        fname = '{}{}'.format(path_images, filename)
        image = io.imread(fname)

        features = rgb_wavelet_features(image)
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

    df = pd.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index=False)

    print 'image feature dataset created.'


def create_feature_dataset(path_images, class_label, path_output,
    f_types=['autocorrelation', 'wavelet'], image_limit=None):

    """
    Create feature vectors from images in directory and save as csv output.

    Parameters
    ----------
    path_images : directory path string
        Directory with images for processing.
    class_label : string
        Class label used in label column of output.
    path_output : directory path string
        Output directory for csv file.
    f_types : array_like, optional
        Specify the feature types:
            - 'autocorrelation'
            - 'wavelet'
        Default: ['autocorrelation', 'wavelet']
    image_limit : int, optional
        Number of images in directory to process.

    Returns
    -------
    csv output file as specified in path_output.

    """

    print 'creating image feature dataset...'

    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        fname = '{}{}'.format(path_images, filename)
        image = io.imread(fname)

        features = {}
        if 'autocorrelation' in f_types:
            lags = ((1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 2))
            features.update(rgb_autocorrelation_features(image, lags))

        if 'wavelet' in f_types:
            features.update(rgb_wavelet_features(image))

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

    df = pd.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index=False)

    print 'image feature dataset created.'


#*******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus/'
    # path_cover = '{}images/train_catdog/cover/'.format(path)
    # path_stego = '{}images/stego/catdog/'.format(path)

    path_cover = '{}images/png/cover/'.format(path)
    path_stego = '{}images/png/stego/'.format(path)

    # autocorrelation_feature_dataset(
    #     path_images=path_cover,
    #     class_label='cover',
    #     path_output='{}data/train_cover_ac.csv'.format(path)
    # )
    #
    # autocorrelation_feature_dataset(
    #     path_images=path_stego,
    #     class_label='stego',
    #     path_output='{}data/train_stego_ac.csv'.format(path)
    # )

    # create_training_set(
    #     '{}data/train_cover_ac.csv'.format(path),
    #     '{}data/train_stego_ac.csv'.format(path),
    #     '{}data/train_ac.csv'.format(path)
    # )

    #***************************************************************************
    # wavelet_feature_dataset(
    #     path_images=path_cover,
    #     class_label='cover',
    #     path_output='{}data/train_cover_wavelet.csv'.format(path)
    # )
    #
    # wavelet_feature_dataset(
    #     path_images=path_stego,
    #     class_label='stego',
    #     path_output='{}data/train_stego_wavelet.csv'.format(path)
    # )
    #
    # create_training_set(
    #     '{}data/train_cover_wavelet.csv'.format(path),
    #     '{}data/train_stego_wavelet.csv'.format(path),
    #     '{}data/train_wavelet.csv'.format(path)
    # )

    #***************************************************************************
    create_feature_dataset(
        path_images=path_cover,
        class_label='cover',
        path_output='{}data/train_cover.csv'.format(path),
        f_types=['autocorrelation', 'wavelet']
    )

    create_feature_dataset(
        path_images=path_stego,
        class_label='stego',
        path_output='{}data/train_stego.csv'.format(path),
        f_types=['autocorrelation', 'wavelet']
    )

    create_training_set(
        '{}data/train_cover.csv'.format(path),
        '{}data/train_stego.csv'.format(path),
        '{}data/train.csv'.format(path)
    )
