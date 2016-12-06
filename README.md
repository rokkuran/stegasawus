# Stegasawus
Detecting whether steganography is present in an image using machine learning.
- Generates dataset for training.
- Creates feature vectors from statistical moments of autocorrelation and discrete wavelet decomposition measures.
- Exploratory plots of images and training datasets.
- Preliminary model comparisons.

### Preliminary Results
Model 5-fold cross validation results on 3960 images of cats and dogs with various message sizes embedded - images are converted to strings and embedded using the Least Significant Bit algorithm. Break down of image types below: cover is the original image, lenna16 is the lenna.png cropped to 16x16 pixels, converted to a string and embedded in the image. Similarly, lenna32 is a 32x32 pixel image.

`Counter({'cover': 1000, 'lenna16': 1000, 'lenna32': 999, 'lenna64': 961})`
```
                    acc       precision   recall     f1      roc_auc
classifier                                                           
lr_lbfgs            0.851010   0.889396  0.913034  0.900806  0.795104
svc_linear_default  0.845707   0.886771  0.908273  0.897191  0.788597
lr_lbfgs_default    0.817929   0.830900  0.948015  0.885277  0.698757
svc_linear          0.810606   0.873211  0.871311  0.872040  0.754934
svc_rbf_default     0.741667   0.741667  1.000000  0.851534  0.500000
lda                 0.741162   0.761581  0.948051  0.844451  0.548113
adaboost            0.739394   0.748319  0.977602  0.847540  0.516979
et                  0.737121   0.740464  0.993835  0.848508  0.496917
xgb                 0.727273   0.741264  0.971410  0.840699  0.499232
gbc                 0.727020   0.742245  0.968916  0.840298  0.501521
knn_default         0.725000   0.737212  0.977397  0.840373  0.488698
rf                  0.723737   0.737062  0.975469  0.839529  0.488198
pa                  0.710606   0.774177  0.861596  0.815136  0.570371
pa_default          0.694697   0.797362  0.789577  0.793004  0.607005
gnb                 0.662879   0.731826  0.861300  0.790695  0.477899
svc_rbf             0.627525   0.715687  0.826008  0.766711  0.442010
knn                 0.548737   0.679961  0.739946  0.708390  0.369973
qda                 0.532828   0.704027  0.639706  0.669862  0.433871
rf_default          0.523232   0.671087  0.702232  0.685790  0.356623
et_default          0.490152   0.655479  0.659750  0.657175  0.331982
```

### Future Work
- Run model benchmarks for specific message sizes and embedding generator types.
- Look at model performance for different image types (only cats and dogs at the moment).
- Model persistence for well performing trained models.
- Write a few non-repeating integer sequence generators for use in LSB embedding.
