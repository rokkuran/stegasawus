from stegasawus.features import (
    create_feature_dataset,
    concatenate_feature_sets,
    concat_multiple_feature_sets)

path = '/home/rokkuran/workspace/stegasawus/'

dim = 64
generator = 'identity'
# generator = 'eratosthenes'

path_cover = '{}images/png/cover/'.format(path)
path_stego = '{}images/png/lenna{}_{}/'.format(path, dim, generator)

# use both feature types to create training set
f_types = ['autocorrelation', 'wavelet']

# create cover image training dataset
create_feature_dataset(
  path_images=path_cover,
  class_label='cover',
  path_output='{}data/train_cover.csv'.format(path),
  f_types=f_types
)

# create steganographic image training dataset
create_feature_dataset(
  path_images=path_stego,
  class_label='stego',
  path_output='%sdata/train_stego_lenna%s_%s.csv' % (path, dim, generator),
  f_types=f_types
)

# merge cover and stego images to create complete training set
concatenate_feature_sets(
  '{}data/train_cover.csv'.format(path),
  '{}data/train_stego_lenna{}_{}.csv'.format(path, dim, generator),
  '{}data/train_lenna_{}.csv'.format(path, generator)
)

# combine multiple training training sets together
concat_multiple_feature_sets(
    [
        '{}data/train_cover.csv'.format(path),
        '{}data/train_stego_lenna16_identity.csv'.format(path),
        '{}data/train_stego_lenna32_identity.csv'.format(path),
        '{}data/train_stego_lenna64_identity.csv'.format(path),
    ],
    '{}data/train_lenna_identity.csv'.format(path)
)
