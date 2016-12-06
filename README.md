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
                    acc       precision   recall     f1      roc_auc
classifier                                                           
lr_lbfgs            0.851010   0.889396  0.913034  0.900806  0.795104
svc_linear_default  0.845455   0.885566  0.909194  0.897106  0.786756
svc_rbf_default     0.741667   0.741667  1.000000  0.851534  0.500000
lda                 0.741162   0.761581  0.948051  0.844451  0.548113
adaboost            0.739394   0.748319  0.977602  0.847540  0.516979
et                  0.737121   0.740493  0.993884  0.848531  0.496942
rf                  0.731313   0.738908  0.985939  0.844620  0.492969
xgb                 0.727273   0.741264  0.971410  0.840699  0.499232
gbc_default         0.727020   0.742379  0.968592  0.840256  0.501842
knn_default         0.725000   0.737212  0.977397  0.840373  0.488698
pa                  0.717929   0.781395  0.862155  0.818990  0.584964
gnb                 0.662879   0.731826  0.861300  0.790695  0.477899
```

### Future Work
- Run model benchmarks for specific message sizes and embedding generator types.
- Look at model performance for different image types (only cats and dogs at the moment).
- Model persistence for well performing trained models.
- Write a few non-repeating integer sequence generators for use in LSB embedding.
