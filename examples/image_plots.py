from stegasawus import eda

from stegano.lsbset import generators


path = '/home/rokkuran/workspace/stegasawus/'
path_cover = '{}images/png/cover/'.format(path)
path_stego = '{}images/png/lenna64_identity/'.format(path)
# path_stego = '{}images/png/lenna64_eratosthenes/'.format(path)
path_output = '{}output'.format(path)

fname = 'cat.2.png'
z = eda.JointImageAnalyser(path_cover + fname, path_stego + fname)

# plot cover and stego images side by side.
z.plot_images()

# plot difference between cover and stego images.
z.plot_difference()

# plot colour channels of cover and stego images.
z.plot_rgb_components()

# Reveal and show hidden image
z.reveal_image(generators.identity(), show=True)

# Plot wavelet decomposition for a colour channel
eda.plot_wavelet_decomposition(z.I[:, :, 0])

# generate set of histogram/kde plots
eda.generate_feature_distplots(
    filepath_train='{}data/features/train_lenna_identity.csv'.format(path),
    path_output=path_output,
    normalise=False
)

# generate set of histograms
eda.generate_feature_histograms(
    filepath_train='{}data/features/train_lenna_identity.csv'.format(path),
    path_output=path_output,
    bins=50,
    normalise=False
)

eda.generate_feature_kde(
    filepath_train='{}data/features/train_lenna_identity.csv'.format(path),
    path_output=path_output,
    normalise=False
)
