# Stegasawus
Detecting whether steganography is present in an image using machine learning.
- Least significant bit (LSB) embedding functions using various embedding location sequences.
- Generates dataset from set cover images.
- Creates feature vectors from statistical moments of autocorrelation and discrete wavelet decomposition measures.
- Exploratory plots of images and training datasets.
- Preliminary model comparisons.

### Preliminary Results
Model 5-fold cross validation results on 2000 (256x256) images of cats and dogs with various message sizes embedded. Image messages are converted to strings and hidden in a cover image using the LSB algorithm.

Image type counts below. Cover is the original image and 64x64 is the image size that is hidden in a cover image.

`Counter({'16x16': 350, '32x32': 333, '64x64': 317, 'cover': 1000})`

![Classifier Accuracy](https://github.com/rokkuran/stegasawus/blob/master/output/plots/clf_embedding_acc.png)

![Classifier Log Loss](https://github.com/rokkuran/stegasawus/blob/master/output/plots/clf_embedding_ll.png)


### Future Work
- Model benchmarks for different LSB embedding generator types.
- Look at model performance for different image types (only cats and dogs at the moment).
- Model persistence for well performing trained models.
- Improve LSB embedding location sequences.
- Extend/improve features.
- Look at jpg images and embedding in discrete cosine coefficients.
