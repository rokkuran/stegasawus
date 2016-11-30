import numpy
import pandas
import os

from scipy import stats
import matplotlib.pyplot as plt
from skimage import io


def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])


def get_ac_features(I, lags=((1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 2))):
    features = {}
    m, n = I.shape

    feature_functions = (
        ('mean', numpy.mean),
        ('stdev', numpy.std),
        ('skew', stats.skew),
        ('kurtosis', stats.kurtosis))

    for x, y in lags:
        ac = I[x:, y:] * I[:m-x, :n-y]
        ac_diff = I[x:, y:] - ac**0.5
        aca = (I[x:, y:] * I[:m-x, :n-y]) / (I[x:, y:].std() * I[:m-x, :n-y].std())
        features['aca_{}{}'.format(x, y)] = aca.sum()

        for f, fn in feature_functions:
            features['ac_{}{}_{}'.format(x, y, f)] = fn(ac.flatten())
            features['ac_diff_{}{}_{}'.format(x, y, f)] = fn(ac_diff.flatten())

    return features


def create_image_ac_feature_dataset(path_images, class_label, path_output, image_limit=None):
    print 'creating image feature dataset...'
    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        image = io.imread(
            fname='{}{}'.format(path_images, filename),
            as_grey=True
        )
        features = get_ac_features(image)
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
    train_cover = pandas.read_csv(filepath_cover)
    train_stego = pandas.read_csv(filepath_stego)
    train = pandas.concat([train_cover, train_stego])
    train.to_csv(path_output, index=False)
    return train


#*******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus/'
    path_cover = '{}images/train_catdog/cover/'.format(path)
    path_stego = '{}images/stego/catdog/'.format(path)

    create_image_ac_feature_dataset(
        path_images=path_cover,
        class_label='cover',
        path_output='{}data/train_cover.csv'.format(path)
    )

    create_image_ac_feature_dataset(
        path_images=path_stego,
        class_label='stego',
        path_output='{}data/train_stego.csv'.format(path)
    )

    create_training_set(
        '{}data/train_cover.csv'.format(path),
        '{}data/train_stego.csv'.format(path),
        '{}data/train.csv'.format(path)
    )
