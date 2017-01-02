import numpy as np
import pandas as pd
import os
import pywt

from scipy import stats
import matplotlib.pyplot as plt
from skimage import io


# ******************************************************************************
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
    """
    Adds prefix to dict keys.

    Parameters
    ----------
    d : dict
        Dictionary to modify.
    prefix : string
        Prefix to add to dict key.

    Returns
    -------
    Modified dict d.

    """
    return {'{}_{}'.format(prefix, k): v for k, v in d.items()}


def autocorrelation_features(I, lags=((1, 0), (0, 1), (1, 1))):
    """
    Calculate the autocorrelation statistical features (specified in
    statistical_metrics function) from a 2D image array for the specified lags.

    Parameters
    ----------
    I : numpy.ndarray
        Array from a greyscale image or an individual colour channel.
    lags : array_like, default : ((1, 0), (0, 1), (1, 1))
        Pixel vertical and horizontal coordinate shift lags.
            e.g. [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]

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


def rgb_autocorrelation_features(I, lags=((1, 0), (0, 1), (1, 1))):
    """
    Calculate the autocorrelation statistical features of a RGB image
    array (m, n, 3) for the specified lags.

    Parameters
    ----------
    I : array
        RGB image array (m, n, 3).
    lags : array_like, default : ((1, 0), (0, 1), (1, 1))
        Pixel vertical and horizontal coordinate shift lags.
            e.g. [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]

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


def concatenate_feature_sets(filepath_cover, filepath_stego, filepath_output):
    """
    Concatenates two feature csv files.

    Parameters
    ----------
    filepath_cover : string
        Filepath to cover image feature set.
    filepath_stego : string
        Filepath to steganographic image feature set.
    filepath_output : string
        Output filepath.

    Returns
    -------
    Concatenated dataset.

    """
    train_cover = pd.read_csv(filepath_cover)
    train_stego = pd.read_csv(filepath_stego)
    train = pd.concat([train_cover, train_stego])
    train.to_csv(filepath_output, index=False)
    return train


def concat_multiple_feature_sets(filepaths, filepath_output):
    train = pd.DataFrame()
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df['filename'] = filepath.split('/')[-1]
        train = pd.concat([train, df])
    train.to_csv(filepath_output, index=False)
    return train


def apply_tolerance(x, tol):
    """
    Applies absolute value filter for given tolerance.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    tol : int, float
        Tolerance.

    Returns
    -------
    Filtered array where |x| >= tol.
    If no values are above the tolerance np.array([0]) is returned.

    """
    x_tol = abs(x) >= tol
    if x_tol.any():
        return x[x_tol]
    else:
        return np.zeros(1)


def wavdec_features(coeffs, tol=1):
    """
    Calculated the statistical features on the components of a mulitlevel 2D
    discrete wavelet decomposition.

    Parameters
    ----------
    coeffs : list
        n level coefficients from pywt.wavedec2
        [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)]
    tol : int, float, default : 1
        Tolerance to apply to individual coefficient arrays.

    Returns
    -------
    features : dict
        Feature vector of statistical components in dictionary form.

    """
    n_layers = len(coeffs) - 1

    features = {}

    cA = coeffs[0]
    prefix = 'dwt_{}_cA'.format(n_layers)
    cA = apply_tolerance(cA, tol)  # reduce sensitivity to noise
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
    """
    For each RGB channel, calculates the statistical features the components of
    a mulitlevel 2D discrete wavelet decomposition.

    Parameters
    ----------
    I : numpy.ndarray
        RGB image array.
    tol : int, float, default : 1
        Tolerance to apply to individual coefficient arrays.

    Returns
    -------
    features : dict
        Feature vector of statistical components in dictionary form.

    """
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        coeffs = pywt.wavedec2(I[:, :, c], wavelet='haar', level=3)
        f_wavelet = wavdec_features(coeffs)
        f_wavelet = prefix_dict_keys(f_wavelet, colour)
        features.update(f_wavelet)

    return features


def create_feature_dataset(path_images, metalabels, path_output,
                           f_types=['autocorrelation', 'wavelet'],
                           image_limit=None):
    """"""
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

        labels = [metalabels[x] for x in sorted(metalabels.keys())]
        row = [filename] + labels
        for feature in feature_names:
            row.append(features[feature])
        dataset.append(row)

        if i % 250 == 0:
            print '{} images processed'.format(i)

        if image_limit:
            if i > image_limit:
                break

    print '{} images processed'.format(i)

    cols = ['filename'] + sorted(metalabels.keys()) + feature_names
    df = pd.DataFrame(dataset, columns=cols)
    df.to_csv(path_output, index=False)
    print 'image feature dataset created.'


if __name__ == '__main__':
    pass
