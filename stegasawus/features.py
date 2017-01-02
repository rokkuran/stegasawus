import numpy as np
import pandas as pd
import os
import pywt

from scipy import stats
import matplotlib.pyplot as plt
from skimage import io



def statistical_metrics(x):
    """
    Calculates statistical metrics on input array (mean, std, skew, kurtosis).
    """

    metrics = {
        'mean': np.mean,
        'stdev': np.std,
        'skew': stats.skew,
        'kurtosis': stats.kurtosis
    }
    return {k: fn(x.flatten()) for k, fn in metrics.items()}


def prefix_dict_keys(d, prefix):
    """Adds prefix to dict keys."""
    return {'{}_{}'.format(prefix, k): v for k, v in d.items()}


def autocorrelation_features(I, lags=((1, 0), (0, 1), (1, 1))):
    """
    Calculate the autocorrelation statistical features (specified in
    statistical_metrics function) from a 2D image array for the specified lags.
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
    """
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        f_ac = autocorrelation_features(I[:, :, c], lags)
        f_ac = prefix_dict_keys(f_ac, colour)
        features.update(f_ac)

    return features


def concatenate_feature_sets(filepath_cover, filepath_stego, filepath_output):
    """Concatenates two feature csv files."""
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
    """Applies absolute value filter for given tolerance."""
    x_tol = abs(x) >= tol
    if x_tol.any():
        return x[x_tol]
    else:
        return np.zeros(1)


def wavdec_features(coeffs, tol=1):
    """
    Calculated the statistical features on the components of a mulitlevel 2D
    discrete wavelet decomposition.
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
    """
    features = {}
    m, n, _ = I.shape

    for c, colour in enumerate('rgb'):
        coeffs = pywt.wavedec2(I[:, :, c], wavelet='haar', level=3)
        f_wavelet = wavdec_features(coeffs)
        f_wavelet = prefix_dict_keys(f_wavelet, colour)
        features.update(f_wavelet)

    return features




class Features(object):
    """"""
    def __init__(self, path_images, path_output):
        super(Features, self).__init__()
        self.path_images = path_images
        self.path_output = path_output

    def create_feature_dataset(self, metalabels,
                               f_types=['autocorrelation', 'wavelet'],
                               image_limit=None):
        """"""
        print 'creating image feature dataset...'

        dataset = list()
        for i, filename in enumerate(os.listdir(self.path_images)):
            fname = '{}{}'.format(self.path_images, filename)
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
        df.to_csv(self.path_output, index=False)
        print 'image feature dataset created.'


if __name__ == '__main__':
    pass
