import numpy
import pandas
from scipy import stats
import matplotlib.pyplot as plt
from skimage import io
from random_lsb_embedding import lsbembed, lsbextract


def rgb_to_grey(image):
    return numpy.dot(image, [0.2989, 0.5870, 0.1140])

def ishow(I, title='', cmap=None):
    plt.imshow(I, cmap=cmap)
    plt.grid(False)
    plt.colorbar()
    if len(title) > 0:
        plt.title(title)
    plt.show()

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


def ac_from_saved_image():
    path = '/home/rokkuran/workspace/stegasawus'
    path_cover = '{}/images/train_catdog/cover/'.format(path)
    path_stego = '{}/images/train_catdog/stego/'.format(path)

    filename = 'cat.698.jpg'
    # filename = 'cat.700.jpg'
    I = rgb_to_grey(io.imread('{}{}'.format(path_cover, filename))).astype(int)
    S = io.imread('{}{}'.format(path_stego, filename))

    # x, y = (1, 0)
    x, y = (0, 1)
    m, n = I.shape
    ishow(numpy.concatenate((I, S), axis=1), title='cover image I; stego image S')
    ishow(abs(I - S), title='Difference: I - S')

    Iac = I[x:, y:] * I[:m-x, :n-y]
    Sac = S[x:, y:] * S[:m-x, :n-y]

    Ip = numpy.concatenate((I[x:, y:], Iac**0.5, I[x:, y:] - Iac**0.5), axis=1)
    Sp = numpy.concatenate((S[x:, y:], Sac**0.5, S[x:, y:] - Sac**0.5), axis=1)
    ishow(Ip)
    ishow(Sp)
    plot_title = 'I, Iac**0.5, I-Iac**0.5\nS, Sac**0.5, S-Sac**0.5\n'
    ishow(numpy.concatenate((Ip, Sp), axis=0), title=plot_title)

    ishow(Iac, 'Iac')
    ishow(Sac, 'Sac')
    ishow(numpy.concatenate((Iac, Sac), axis=1), title='Iac and Sac')

    Ai = 2*(Iac%2).astype(int)-1
    As = 2*(Sac.astype(int)%2).astype(int)-1
    ishow(numpy.concatenate((Ai, As), axis=1), title='Autocorrelation of I and S')


def ac_from_in_memory_image():
    path = '/home/rokkuran/workspace/stegasawus'
    path_cover = '{}/images/train_catdog/cover/'.format(path)

    filename = 'cat.698.jpg'
    # filename = 'cat.700.jpg'
    I = rgb_to_grey(io.imread('{}{}'.format(path_cover, filename))).astype(int)
    S = lsbembed(I, message='secret '*1000)
    io.imsave('{}/images/steg_{}'.format(path, filename), S)
    Sr = io.imread('{}/images/steg_{}'.format(path, filename))

    x, y = (1, 0)
    m, n = I.shape
    ishow(numpy.concatenate((I, S), axis=1), title='cover image I; stego image S')
    ishow(I - S, title='Difference: I - S')

    Iac = I[x:, y:] * I[:m-x, :n-y]
    Sac = S[x:, y:] * S[:m-x, :n-y]

    ishow(numpy.concatenate((Iac, Sac), axis=1), title='Iac and Sac')

    Ai = 2*(Iac%2).astype(int)-1
    As = 2*(Sac%2).astype(int)-1
    ishow(numpy.concatenate((Ai, As), axis=1), title='Autocorrelation of I and S')


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


path = '/home/rokkuran/workspace/stegasawus'
# path_cover = '{}/images/train_catdog/cover/'.format(path)
# path_stego = '{}/images/train_catdog/stego/'.format(path)

path_cover = '{}/images/train_catdog_rndembed/cover/'.format(path)
path_stego = '{}/images/train_catdog_rndembed/stego/'.format(path)

# path_cover = '{}/images/train/cropped/'.format(path)
# path_stego = '{}/images/train/encoded/'.format(path)

create_image_ac_feature_dataset(
    path_images=path_cover,
    class_label='cover',
    # path_output='{}/data/train_catdog_ac_cover.csv'.format(path)
    path_output='{}/data/train_catdog_rndembed_ac_cover.csv'.format(path)
    # path_output='{}/data/train_ac_cover.csv'.format(path)
)

create_image_ac_feature_dataset(
    path_images=path_stego,
    class_label='stego',
    # path_output='{}/data/train_catdog_ac_stego.csv'.format(path)
    path_output='{}/data/train_catdog_rndembed_ac_stego.csv'.format(path)
    # path_output='{}/data/train_ac_stego.csv'.format(path)
)

# create_training_set(
#     '{}/data/train_catdog_ac_cover.csv'.format(path),
#     '{}/data/train_catdog_ac_stego.csv'.format(path),
#     '{}/data/train_catdog_ac.csv'.format(path)
# )

create_training_set(
    '{}/data/train_catdog_rndembed_ac_cover.csv'.format(path),
    '{}/data/train_catdog_rndembed_ac_stego.csv'.format(path),
    '{}/data/train_catdog_rndembed_ac.csv'.format(path)
)

# create_training_set(
#     '{}/data/train_ac_cover.csv'.format(path),
#     '{}/data/train_ac_stego.csv'.format(path),
#     '{}/data/train_ac.csv'.format(path)
# )
