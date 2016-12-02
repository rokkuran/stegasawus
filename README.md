# Stegasawus
Detecting whether steganography is present in an image using machine learning.

## Data
- Create datasets of cover (original) and steganographic (hidden message) images using Least Significant Bit (LSB) embedding. Uses [Stegano][stegano] package for embedding.
-

## Features
Feature vectors created for each RGB colour channel using the statistical moments (mean, var, skew, kurtosis) of autocorrelation and discrete wavelet decomposition results.

## Analysis
- Image specific plots of cover/stego image
- Plots of feature distributions: histograms, kernel density estimation.

### Preliminary Results
Mean 10 fold cross validation results using image of cropped (256x256) cat and dog images

### Future Work
- Run model benchmarks for specific message sizes and embedding generator types.
- Look at model performance for different image datasets types.

[stegano]: https://www.mozilla.org
