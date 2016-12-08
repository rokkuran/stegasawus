# Stegasawus
Detecting whether steganography is present in an image using machine learning.
- Generates dataset for training.
- Creates feature vectors from statistical moments of autocorrelation and discrete wavelet decomposition measures.
- Exploratory plots of images and training datasets.
- Preliminary model comparisons.

### Preliminary Results
Model 5-fold cross validation results on 3960 (dimensions=256x256) images of cats and dogs with various message sizes embedded - images are converted to strings and embedded using the Least Significant Bit algorithm. Break down of image types below: cover is the original image, lenna16 is the lenna.png cropped to 16x16 pixels, converted to a string and embedded in the image. Similarly, lenna32 is a 32x32 pixel image.

`Counter({'cover': 1000, 'lenna16': 1000, 'lenna32': 999, 'lenna64': 961})`

```
                    acc    log_loss   precision  recall     f1      roc_auc
classifier                                                                  
svc_linear        0.8370   5.629867   0.868915  0.788451  0.826650  0.836486
lr_lbfgs          0.8285   5.923461   0.839406  0.807798  0.823155  0.828125
lr_lbfgs_default  0.7885   7.305007   0.835395  0.713187  0.768811  0.788010
lda               0.7520   8.565647   0.883020  0.576351  0.695906  0.750343
pa                0.6470  12.192319   0.649203  0.616547  0.631840  0.645586
pa_default        0.6465  12.209611   0.635175  0.674285  0.652296  0.648117
rf                0.6280  12.848548   0.646349  0.562253  0.597762  0.629976
gnb               0.5720  14.782827   0.551631  0.717321  0.623072  0.573963

```

![Classifier Accuracy](https://github.com/rokkuran/stegasawus/blob/master/output/plots/clf_embedding_acc.png)

![Classifier Log Loss](https://github.com/rokkuran/stegasawus/blob/master/output/plots/clf_embedding_ll.png)

### Future Work
- Run model benchmarks for specific message sizes and embedding generator types.
- Look at model performance for different image types (only cats and dogs at the moment).
- Model persistence for well performing trained models.
- Write a few non-repeating integer sequence generators for use in LSB embedding.
- Write own LSB embedding algorithm.
