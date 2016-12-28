from stegasawus import lsb, seq

import os
import base64
import cStringIO
from PIL import Image

from os import path, listdir
from skimage import io


def get_secret_message(filepath):
    """
    Read text file, return message.
    """
    with open(filepath, 'rb') as f:
        message = f.read()
    return message


def image_to_string(path_image):
    with open(path_image, 'rb') as f:
        return base64.b64encode(f.read())


def string_to_image(image_string):
    s = base64.b64decode(image_string)
    s = cStringIO.StringIO(s)
    return np.array(Image.open(s))


def crop_image(image, dim, centre=True):
    m, n = dim
    if centre:
        x, y, _ = image.shape
        x0 = int((x - m) / 2) - 1
        y0 = int((y - n) / 2) - 1
        xm = int(m + x0)
        yn = int(n + y0)
        return image[x0:xm, y0:yn]
    else:
        return image[0:m, 0:n]


def crop_images(path_images, path_output, dimensions, centre=True):
    """
    Batch crop images from top left hand corner to dimensions specified. Skips
    images where dimensions are incompatible.
    """
    print 'cropping images...'
    for i, filename in enumerate(os.listdir(path_images)):
        try:
            image = io.imread('{}{}'.format(path_images, filename))
            cropped = crop_image(image, dimensions, centre=centre)
            io.imsave(
                fname='{}{}'.format(path_output, filename),
                arr=cropped
            )
            print '{}: {}'.format(i, filename)
        except IndexError:
            print '{}: {} failed - dimensions incompatible'.format(i, filename)

    print 'all images cropped and saved.'


def batch_jpg_to_png(path_input, path_output):
    """
    Convert jpg images to png.
    """
    print 'coverting images...'
    for i, filename in enumerate(os.listdir(path_input)):
        input_jpg = '{}{}'.format(path_input, filename)

        fname = filename.replace('.jpg', '.png')
        output_png = '{}{}'.format(path_output, fname)

        I = io.imread(input_jpg)
        io.imsave(output_png, I)
        print '{}: {}'.format(i, filename)
    print 'image conversion complete.'


class DatasetGenerator(object):
    """
    Generates dataset from
    """
    def __init__(self, path_images, path_output, seq_method):
        super(DatasetGenerator, self).__init__()
        self._path_images = path_images
        self._path_output = path_output
        self._seq_method = seq_method

    def _read_embed_save(self, filename, message):
        try:
            path_cover = '{}{}'.format(self._path_images, filename)
            path_stego = '{}{}'.format(self._path_output, filename)
            I = io.imread(path_cover)
            S = lsb.embed(I, message, self._seq_method)
            io.imsave(arr=S, fname=path_stego)
        except KeyError as e:
            print '%s | message size greater than image capacity.' % filename

    def batch_hide_message(self, message):
        # TODO: cleanup error handling
        for i, filename in enumerate(listdir(self._path_images), start=1):
            file_type = filename.split('.')[-1]
            if file_type == 'png':
                self._read_embed_save(filename, message)
                print '{}: {}'.format(i, filename)
            else:
                error = 'Image type not supported. Supported types: {png}'
                raise Exception(error)

        print 'image encoding complete.'


def create_cropped_set(path_image, path_output, dims):
    I = io.imread(path_image)
    for m, n in dims.items():
        pass


if __name__ == '__main__':
    cdir = '/home/rokkuran/workspace/stegasawus/'
    fp = path.join(cdir, 'data/messages/Lenna_64x64.png')

    msg = image_to_string(fp)

    path_images = '{}images/png/cover_test/'.format(cdir)
    path_output = '{}images/png/lsb_test/'.format(cdir)

    seq_method = seq.rand_darts(seed=77)
    g = DatasetGenerator(path_images, path_output, seq_method)
    g.batch_hide_message(msg)
