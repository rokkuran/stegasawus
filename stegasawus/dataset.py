import os

from skimage import io

from stegano import exifHeader
from stegano import lsbset
from stegano.lsbset import generators


# ******************************************************************************
def get_secret_message(filepath):
    """
    Read text file.

    Parameters
    ----------
    filepath : string
        Input file.

    Returns
    -------
    message : string
        Contents of file.

    """
    with open(filepath, 'rb') as f:
        message = f.read()
    return message


def hide_message_jpg(secret_message, cover_file, stego_file):
    """
    Hide message in jpg file.

    Parameters
    ----------
    secret_message : string
        Message to hide.
    cover_file : string, filepath
        Input image to hide message in.
    stego_file : string, filepath
        Output image with hidden message embedded.

    """
    exifHeader.hide(cover_file, stego_file, secret_message=secret_message)


def batch_hide_message(secret_message, path_images, path_output, file_type,
                       generator=''):
    """
    Create steganographic images containing the hidden message using the Least
    Significant Bit (LSB) algorithm.

    Parameters
    ----------
    secret_message : string
        Hidden message to embed in image.
    path_images : string
        Input image directory.
    path_output : string
        Output directroy for steganographic images.
    file_type : string
        'png' - embeds message in rgb pixels.
        'jpg' - embeds message in discrete cosine coefficients.
    generator : string, optional
        Specify the embedding location generator.

    Returns
    -------
    None
        Steganographic images saved in path_output.

    """

    print 'encoding images...'
    file_type = file_type.lower()

    for i, filename in enumerate(os.listdir(path_images), start=1):
        try:
            cover = '{}{}'.format(path_images, filename)
            stego = '{}{}'.format(path_output, filename)

            if file_type == 'png':
                S = lsbset.hide(
                    cover,
                    secret_message=secret_message,
                    generator=generators.identity()
                )
                S.save(stego)
            elif file_type == 'jpg':
                exifHeader.hide(cover, stego, secret_message=secret_message)
            else:
                print "file_type '%s' is not supported. \noptions = jpg, png."

            print '{}: {}'.format(i, filename)
        except IndexError as e:
            print '{}: {} | IndexError: {}'.format(i, filename, e)
        except Exception as e:
            print '{} : {} | Exception: {}'.format(i, filename, e)

    print 'image encoding complete.'


def batch_hide_message_rnd_generator():
    pass


def crop_images(path_images, path_output, dimensions=(256, 256)):
    """
    Batch crop images from top left hand corner to dimensions specified. Skips
    images where dimensions are incompatible.

    Parameters
    ----------
    path_images : string, directroy path
        Directory with image set to crop.
    path_output : string, directroy path
        Directory for output of cropped images.
    dimensions : array_like
        Dimensions to crop image to: (n, m) array.

    """
    print 'cropping images...'
    m, n = dimensions
    for i, filename in enumerate(os.listdir(path_images)):
        try:
            image = io.imread('{}{}'.format(path_images, filename))
            cropped = image[0:m, 0:n]
            io.imsave(
                fname='{}{}'.format(path_output, filename),
                arr=cropped
            )
            print '{}: {}'.format(i, filename)
        except IndexError:
            print '{}: {} failed - dimensions incompatible'.format(i, filename)

    print 'all images cropped and saved.'


def batch_png_to_jpg(path_input, path_output):
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


# ******************************************************************************
if __name__ == '__main__':
    pass
