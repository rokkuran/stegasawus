import os
import numpy
import pandas
import pywt

import skimage
import skimage.io as io

from skimage import transform
from scipy import stats

import matplotlib.pyplot as plt


def calc_pyramid(image, max_layer=3, downscale=2):
    pyramid = tuple(transform.pyramid_gaussian(
        image=image,
        max_layer=max_layer,
        downscale=2
    ))
    return pyramid

def calc_pyramid_residuals(pyramid):
    """
    First layer in pyramid is original image. Calculate residuals between
    filtered downsampled images subsequently rescaled and the original.
    """
    residuals = []
    for x in pyramid[1:]:
        x_resized = transform.resize(x, pyramid[0].shape)
        residuals.append(pyramid[0] - x_resized)
    return residuals

def plot_pyramid_residuals(arg):
    # original then 3 residuals
    pass

def get_feature_vector(a):
    feature_functions = [
        ('mean', numpy.mean),
        ('stdev', numpy.std),
        ('skew', stats.skew),
        ('kurtosis', stats.kurtosis)
        # ('entropy', stats.entropy)
    ]
    feature_names = zip(*feature_functions)[0]

    feature_vector = []
    for (feature, fn) in feature_functions:
        feature_vector.append(fn(a.flatten()))

    return feature_names, feature_vector

def get_pyramid_features(pyramid, pyramid_type):
    features = {}
    for i, layer in enumerate(pyramid):
        feature_names, feature_vector = get_feature_vector(layer)
        for metric, value in zip(feature_names, feature_vector):
            feature_name = '{}_{}_{}'.format(pyramid_type, i, metric)
            features[feature_name] = value
    return features

def gaussian_pyramid_features(image, max_layer=3, downscale=2):
    pyramid = calc_pyramid(image, max_layer, downscale)
    residuals = calc_pyramid_residuals(pyramid)

    features = get_pyramid_features(pyramid, pyramid_type='gp')
    features.update(get_pyramid_features(residuals, pyramid_type='gp_res'))

    return features

def get_image_features(image):
    features = gaussian_pyramid_features(image)
    # features.update(other_features_here)
    return features

def create_image_feature_dataset(path_images, class_label, path_output, image_limit=None):
    print 'creating image feature dataset...'
    dataset = list()
    for i, filename in enumerate(os.listdir(path_images)):
        image = io.imread(
            fname='{}{}'.format(path_images, filename),
            as_grey=True
        )
        features = get_image_features(image)
        if i == 0:
            feature_names = features.keys()

        row = [filename, class_label]
        for feature in feature_names:
            row.append(features[feature])

        dataset.append(row)

        if i % 250 == 0:
            print '{} images processed'.format(i)

        if image_limit:
            if i > image_limit:
                break

    df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    df.to_csv(path_output, index=False)
    print 'image feature dataset created.'


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'

    def test_feature_generation():
        path_images = '{}/images/originals/'.format(path)
        filename = '19694.jpg'

        image = io.imread(
            fname='{}{}'.format(path_images, filename),
            as_grey=True
        )
        features = gaussian_pyramid_features(image)
        return features

    # path_cropped = '{}/images/train/cropped/'.format(path)
    # create_image_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='clean',
    #     path_output='{}/data/train_cropped.csv'.format(path)
    # )
    #
    # path_encoded = '{}/images/train/encoded/'.format(path)
    # create_image_feature_dataset(
    #     path_images=path_cropped,
    #     class_label='message',
    #     path_output='{}/data/train_encoded.csv'.format(path)
    # )

    def create_training_set(filepath_cover, filepath_stego, path_output):
        train_cover = pandas.read_csv(filepath_cover)
        train_stego = pandas.read_csv(filepath_stego)
        train = pandas.concat([train_cover, train_stego])
        train.to_csv(path_output, index=False)
        return train

    # create_training_set('{}/data/train.csv'.format(path))

    def plot_dwt(image):
        cA, cD = pywt.dwt(image, 'haar')
        print 'shape | image {}; cA {}; cD {}'.format(image.shape, cA.shape, cD.shape)
        plot_image = numpy.concatenate((cA, cD), axis=1)
        plt.imshow(plot_image)
        plt.show()

    def plot_dwt2(image):
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        print 'shape | image {}; cA {}; cD {}'.format(image.shape, cA.shape, cD.shape)
        cAcH = numpy.concatenate((abs(cA), abs(cH)), axis=1)
        cVcD = numpy.concatenate((abs(cV), abs(cD)), axis=1)
        plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
        plt.imshow(plot_image)
        plt.show()

    def plot_dwt_coefficients(coeffs):
        cA, (cH, cV, cD) = coeffs
        print 'shape | cA {}; cH {}; cV {}; cD {}'.format(
            cA.shape, cH.shape, cV.shape, cD.shape
        )
        cAcH = numpy.concatenate((cA, cH), axis=1)
        cVcD = numpy.concatenate((cV, cD), axis=1)
        plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
        plt.imshow(plot_image)
        plt.show()

    def dwt_levels(image):
        coeffs = pywt.wavedec2(image, wavelet='haar', level=3)
        return coeffs

    # path_output = '{}/images/'.format(path)
    # filename = '18_1.jpg'
    # # filename = '17_1.jpg'
    # image = io.imread(fname='{}{}'.format(path_output, filename), as_grey=True)
    # io.imshow(image)
    # plt.show()
    # # plot_dwt(image)
    # # plot_dwt2(image)

    def rgb_to_grey(image):
        return numpy.dot(image, [0.2989, 0.5870, 0.1140])

    def plot_wavelet_decomposition(image, coeffs):
        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            if i == 0:
                cAcH = numpy.concatenate((coeffs[0], cH), axis=1)
                cVcD = numpy.concatenate((cV, cD), axis=1)
                plot_image = numpy.concatenate((cAcH, cVcD), axis=0)
            else:
                plot_image = numpy.concatenate((plot_image, cH), axis=1)
                cVcD = numpy.concatenate((cV, cD), axis=1)
                plot_image = numpy.concatenate((plot_image, cVcD), axis=0)

        # plot_image = pywt.coeffs_to_array(coeffs)
        plt.grid(False)
        io.imshow(plot_image)#, cmap='gray')
        plt.show()
    #
    # path_output = '{}/images/'.format(path)
    # filename = '17_11.jpg'
    # # image = io.imread(fname='{}{}'.format(path_output, filename), as_grey=True)
    # image = io.imread(fname='{}{}'.format(path_output, filename))[:, :, 0]
    # coeffs = pywt.wavedec2(image, wavelet='haar', level=3)
    # plot_wavelet_decomposition(image, coeffs)

    def create_feature_name(layer, c, fname):
        return 'dwt_{layer}_{c}_{fname}'.format(layer=layer, c=c, fname=fname)


    def get_wavdec_feature_vector(coeffs):
        feature_functions = [
            ('mean', numpy.mean),
            ('stdev', numpy.std),
            ('skew', stats.skew),
            ('kurtosis', stats.kurtosis)]

        feature_vector = {}
        cA = coeffs[0]
        for (fname, fn) in feature_functions:
            feature_name = 'dwt_{layer}_cA_{fname}'.format(
                layer=len(coeffs) - 1, fname=fname
            )
            # reduce sensitivity to noise
            c_tol = abs(cA) > 1 # coefficients with magnitude > 1 allowed
            if c_tol.any():
                feature_vector[feature_name] = fn(cA[c_tol].flatten())
            else:
                feature_vector[feature_name] = 0

        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            layer = len(coeffs) - 1 - i

            for (fname, fn) in feature_functions:
                for c, cX in zip(('cH', 'cV', 'cD'), (cH, cV, cD)):
                    feature_name = 'dwt_{layer}_{c}_{fname}'.format(
                        layer=layer, c=c, fname=fname
                    )
                    c_tol = abs(cX) > 1
                    if c_tol.any():
                        feature_vector[feature_name] = fn(cX[c_tol].flatten())
                    else:
                        feature_vector[feature_name] = 0

        return feature_vector

    # feature_vector = get_wavdec_feature_vector(coeffs)


    # def create_image_wavdec_feature_dataset(path_images, class_label, path_output, image_limit=None):
    #     print 'creating image feature dataset...'
    #     dataset = list()
    #     for i, filename in enumerate(os.listdir(path_images)):
    #         image = io.imread(
    #             fname='{}{}'.format(path_images, filename)
    #             # as_grey=True
    #         )
    #         if len(image.shape) == 3:   # make sure rgb image arrays
    #             coeffs = pywt.wavedec2(image[:, :, 0], wavelet='haar', level=3)
    #             features = get_wavdec_feature_vector(coeffs)
    #             if i == 0:
    #                 feature_names = features.keys()
    #
    #             row = [filename, class_label]
    #             for feature in feature_names:
    #                 row.append(features[feature])
    #
    #             dataset.append(row)
    #
    #         if i % 250 == 0:
    #             print '{} images processed'.format(i)
    #
    #         if image_limit:
    #             if i > image_limit:
    #                 break
    #
    #     df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
    #     df.to_csv(path_output, index=False)
    #     print 'image feature dataset created.'


    def create_image_wavdec_feature_dataset(path_images, class_label, path_output, image_limit=None):
        print 'creating image feature dataset...'
        dataset = list()
        for i, filename in enumerate(os.listdir(path_images)):
            image = io.imread(fname='{}{}'.format(path_images, filename))
            # print image.shape
            try:
                if len(image.shape) == 3:   # make sure rgb image arrays
                    image = rgb_to_grey(image)
                    dataset.append([filename, class_label] + farid36(image) + farid36pred(image))
                else:
                    dataset.append([filename, class_label] + farid36(image) + farid36pred(image))
            except ValueError as e:
                print i, filename, e
            except Exception as e:
                print i, filename, e
                raise

            if i % 250 == 0:
                print '{} images processed'.format(i)

            if image_limit:
                if i > image_limit:
                    break

        feature_names = []
        metrics = ['mean', 'stdev', 'skew', 'kurtosis']
        for i in xrange(1, 4):
            for c in ['cH', 'cV', 'cD', 'cHp', 'cVp', 'cDp']:
                for m in metrics:
                    feature_names.append('farid_{}_{}_{}'.format(i, c, m))

        df = pandas.DataFrame(dataset, columns=['image', 'label'] + feature_names)
        df.to_csv(path_output, index=False)
        print 'image feature dataset created.'

#*******************************************************************************

    path_images = '/home/rokkuran/workspace/kaggle/cats_vs_dogs/train/cats/'
    path_cover = '{}/images/train_catdog/cover/'.format(path)
    path_stego = '{}/images/train_catdog/stego/'.format(path)


    path_cropped = '{}/images/train/cropped/'.format(path)
    create_image_wavdec_feature_dataset(
        # path_images=path_cropped,
        path_images=path_cover,
        class_label='cover',
        # path_output='{}/data/train_cropped.csv'.format(path)
        path_output='{}/data/train_catdog_cover.csv'.format(path)
    )

    path_encoded = '{}/images/train/encoded/'.format(path)
    create_image_wavdec_feature_dataset(
        # path_images=path_cropped,
        path_images=path_stego,
        class_label='stego',
        # path_output='{}/data/train_encoded.csv'.format(path)
        path_output='{}/data/train_catdog_stego.csv'.format(path)
    )

    # create_training_set(
    #     '{}/data/train_cropped.csv'.format(path),
    #     '{}/data/train_encoded.csv'.format(path),
    #     '{}/data/train.csv'.format(path)
    # )

    create_training_set(
        '{}/data/train_catdog_cover.csv'.format(path),
        '{}/data/train_catdog_stego.csv'.format(path),
        '{}/data/train_catdog.csv'.format(path)
    )

#*******************************************************************************

    from scipy import sqrt,ceil
    from pywt import Wavelet


    def getWavelet(name):
      if name[:3] == "qmf": return qmfWavelet(name)
      else: return pywt.Wavelet(name)


    QMFtable = {
      "qmf5" : [ -0.076103, 0.3535534, 0.8593118, 0.3535534, -0.076103 ],
      "qmf9" : [ 0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934,
              0.41472545, -0.073386624, -0.060944743, 0.02807382 ],
      "qmf13" : [ -0.014556438, 0.021651438, 0.039045125, -0.09800052,
              -0.057827797, 0.42995453, 0.7737113, 0.42995453, -0.057827797,
              -0.09800052, 0.039045125, 0.021651438, -0.014556438 ],
      "qmf8" : map( lambda x : x*sqrt(2),
                [0.00938715, -0.07065183, 0.06942827, 0.4899808,
                0.4899808, 0.06942827, -0.07065183, 0.00938715 ] ),
      "qmf12" : map( lambda x : x*sqrt(2),
         [-0.003809699, 0.01885659, -0.002710326, -0.08469594,
           0.08846992, 0.4843894, 0.4843894, 0.08846992, -0.08469594,
           -0.002710326, 0.01885659, -0.003809699 ] ),
      "qmf16" : map( lambda x : x*sqrt(2),
               [0.001050167, -0.005054526, -0.002589756, 0.0276414,
                -0.009666376, -0.09039223, 0.09779817, 0.4810284, 0.4810284,
                0.09779817, -0.09039223, -0.009666376, 0.0276414, -0.002589756,
                -0.005054526, 0.001050167 ] ),
    }

    def hfilt(filt):
      sz = len(filt)
      sz2 = ceil(float(sz)/2) - 1
      h = [ filt[i]*(-1)**(i-sz2) for i in xrange(sz-1,-1,-1) ]
      return h

    def rev(filt):
      """
        Reverse the order of a list.
        This is a functional alternative to list.reverse() which
        reverses the list in place.  The functional approach means
        that the reversed list can be used in an expression without
        explicitely being copied or assigned to a name.
      """
      sz = len(filt)
      return [ filt[i] for i in xrange(sz-1,-1,-1) ]

    def qmfWavelet(name):
      """
        Return a Wavelet object representing the given QMF wavelet.
      """
      lf = QMFtable[name]
      if len(lf) % 2 == 1: lf = lf + [0]
      hf = hfilt(lf)
      f = ( lf, hf, rev(lf), rev(hf) )
      return Wavelet(name,f)


    from scipy.stats import skew as skewness, kurtosis
    from numpy import mean, var
    from scipy import log2
    import numpy as np
    import pywt

    def reshape(A):
      return np.reshape(np.transpose(A),(A.size,1))

    def moments(E):
      """
        Given an array E, calculate mean, variance, skewness and kurtosis.
        (Auxiliary function for the Lyu-Farid feature vector.)
      """
      E = E.flatten()
      return [ mean(E), var(E), skewness(E), kurtosis(E) ]

    def predictionFeatures(L,verbosity=0):
      """
        Compute the linear predictions and its error statistics
        from a series of matrices.  (Auxiliary function for pred1().
      """
      S = [ abs(A.flatten()) for A in L ]
      B = ( S[0] >= 1 )
      S = [ A[B] for A in S ]
      S = [ A[:,None] for A in S ]
      T = S[0]
      Q = np.hstack(S[1:])
      if verbosity > 0:
        print "T", type(T), T.shape, T.dtype
        print "Q", type(Q), Q.shape, Q.dtype

    # W -- Regression weights
    #
    #   ::

      QT = Q.transpose()
      Q2 = np.dot(QT,Q)
      Q2I = np.linalg.inv(Q2)
      Q2 = np.dot( Q2I, QT )
      W = np.dot(Q2, T)

    # P -- Prediction
    #
    #   ::

      P = np.dot(Q, W)

    # E -- Errors
    #
    #   ::

      if (T==0).any() or (P==0).any():
         print "T:", (T==0).any()
         print "P:", (P==0).any()
         raise Exception, "Zero occurs in logarithm"
      E = log2(T) - log2(abs(P))
      if verbosity > 2:
        print E.dtype, T.dtype, P.dtype, Q.dtype, W.dtype
        print np.min(T.flatten()), np.min(P.flatten())
      R = moments(E)
      if reduce( bool.__or__, [ bool(np.isinf(r) or np.isnan(r)) for r in R ] ):
    	print R
    	raise Exception, "NaN or inf detected in Farid feature."
      return R

    # Creating the prediction image
    #
    # ::

    def pred1(hv,verbosity=0,*a,**kw):
      """
        Compute the linear prediction error statistics from one
        level of the wavelet decomposition.
      """
      (H,V,D) = hv[-1]
      (H1,V1,D1) = hv[-2]
      R = []
      (X,Y) = H.shape
      hx = [ i/2 for i in xrange(1,X-1) ]
      hy = [ i/2 for i in xrange(1,Y-1) ]
      D1 = D1[hx,:][:,hy]
      H1 = H1[hx,:][:,hy]
      V1 = V1[hx,:][:,hy]
      if verbosity > 0: print "H",
      R += predictionFeatures( [
          H[1:-1,1:-1], H[0:-2,1:-1], H[1:-1,0:-2],
          H1, D[1:-1,1:-1], D1,
          H[2:,1:-1], H[1:-1,2:] ], verbosity=verbosity )
      if verbosity > 0: print "V",
      R += predictionFeatures( [
          V[1:-1,1:-1], V[0:-2,1:-1], V[1:-1,0:-2],
          V1, D[1:-1,1:-1], D1, V[2:,1:-1],
          V[1:-1,2:] ], verbosity=verbosity )
      if verbosity > 0: print "D",
      R += predictionFeatures( [
          D[1:-1,1:-1], D[0:-2,1:-1], D[1:-1,0:-2], D1,
          H[1:-1,1:-1], V[1:-1,1:-1], D[2:,1:-1], D[1:-1,2:] ],
          verbosity=verbosity )
      return R

    def pred(H,*a,**kw):
      """
        Given a tuple H containing a wavelet decomposition of an image,
        calculate the Lyu-Farid feature based on the linear predictor.
      """
      if len(H) < 3: return []
      else: return pred(H[:-1], *a, **kw) + pred1(H, *a, **kw)

    def fxbase(H,*a,**kw):
      R = []
      for h in H[2:]:
        for A in h:
          R.extend( moments(A) )
      return R

    # The Feature Vector
    # ==================

    def farid36(I,name="qmf9",*a,**kw):
      """
        Calculate the 36 Farid features from the image I,
        excluding the prediction image features.
        Optionally a wavelet name can be given as well.
      """
      w = getWavelet( name )
      I = I.astype( float )
      H = pywt.wavedec2( I, w, level=4 )
      return fxbase(H,*a,**kw)

    def farid36pred(I,name="qmf9",*a,**kw):
      """
        Calculate the 36 Farid features from the prediction image of I.
        Optionally a wavelet name can be given as well.
      """
      w = getWavelet( name )
      I = I.astype( float )
      H = pywt.wavedec2( I, w, level=4 )
      return pred(H,*a,**kw)
