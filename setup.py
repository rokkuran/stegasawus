#!/usr/bin/env python

from distutils.core import setup

setup(
    name='stegasawus',
    version='0.1',
    description='Machine learning detection of steganographic images',
    author='Lachlan Taylor',
    author_email='lachlanbtaylor@gmail.com',
    packages=['stegasawus']
    # data_files=[('./images/', ['Lenna.png', 'image.png'])]
)