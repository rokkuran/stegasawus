from stegasawus.dataset import (
    get_secret_message,
    batch_png_to_jpg,
    crop_images,
    batch_hide_message,
    create_benchmark_image_message
)


# Crop all images in a file and ouput to directory
crop_images(
    path_images='{}images/jpg/cats_and_dogs/original/'.format(path)
    path_output='{}images/jpg/cats_and_dogs/cropped_256/'.format(path),
    dimensions=(256, 256),
    centre=True
)


# Batch convert jpg to png
batch_png_to_jpg(
    path_input='{}images/jpg/cats_and_dogs/cropped_256/'.format(path),
    path_output='{}images/png/cover/'.format(path)
)


# Create steganographic image set embedding the secret message
generator = 'identity'
dim = 64

path = '/home/rokkuran/workspace/stegasawus/'
path_images = '{}images/png/cover/'.format(path)
path_output = '{}images/png/lenna{}_{}/'.format(path, dim, generator)
path_msg = '{}data/messages/'.format(path)

secret_message = get_secret_message(
    '{}Lenna_{}x{}.txt'.format(path_msg, dim, dim)
)

batch_hide_message(
    secret_message=secret_message,
    path_images=path_images,
    path_output=path_output,
    file_type='png',
    generator=generator
)
