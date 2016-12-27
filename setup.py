#!/usr/bin/env python

from distutils.core import setup

setup(
    name='stegasawus',
    version='0.3.8',
    description='Machine learning detection of steganographic images',
    author='Lachlan Taylor',
    packages=['stegasawus']  # 'stegasawus.lsb']
    # data_files=[('./images/', ['Lenna.png', 'image.png'])]
)
