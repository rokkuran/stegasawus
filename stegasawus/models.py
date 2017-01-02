import numpy as np
import pandas as pd
import yaml
import re

import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score,
    roc_auc_score)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, LabelBinarizer, PolynomialFeatures)
from sklearn.model_selection import (
    GridSearchCV, learning_curve, ShuffleSplit, StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, ActivityRegularization
# from keras.regularizers import WeightRegularizer
# from keras.wrappers.scikit_learn import KerasClassifier

# from xgboost import XGBClassifier


input_dim = 125


class ModelComparer(object):
    """"""
    def __init__(self, X, y, pipeline, splitter, classifiers, metrics=None):
        super(ModelComparer, self).__init__()
        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.splitter = splitter
        self.classifiers = classifiers

        if metrics is None:
            self.metrics = [accuracy_score, log_loss, precision_score,
                            recall_score, f1_score, roc_auc_score]
        else:
            self.metrics = metrics

        self._scores = []
        self.models = {}

    def _get_pipeline(self, name):
        return make_pipeline(self.pipeline, classifiers[name])

    @property
    def cv_split_generator(self):
        """
        Train and validation set split generator.
        """
        g = enumerate(self.splitter.split(self.X, self.y))
        for i, (train_idx, val_idx) in g:
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            yield i, X_train, X_val, y_train, y_val

    def _metric_name(self, f):
        return f.__name__.replace('_score', '')

    def _get_metric_scores(self, y_val, y_pred):
        s = str()
        scores = []
        for fn in self.metrics:
            score = fn(y_val, y_pred)
            scores.append(score)
            s += '%s = %.4f; ' % (self._metric_name(fn), score)
        return scores, s

    def model_comparison(self, cv_mean=True):
        self._scores = []
        for i, X_train, X_val, y_train, y_val in self.cv_split_generator:
            for name, clf in self.classifiers.items():
                pipeline = self._get_pipeline(name)
                model = pipeline.fit(X_train, y_train)
                self.models[name] = model

                y_pred = model.predict(X_val)
                metrics, ps = self._get_metric_scores(y_val, y_pred)
                ps += ' | %s_%d' % (name, i)
                print ps
                self._scores.append([name, i] + metrics)

    def scores(self, mean=True):
        cols = ['classifier', 'split']
        cols += [self._metric_name(fn) for fn in self.metrics]

        df = pd.DataFrame(self._scores, columns=cols)
        df = df.sort_values(
            by=['accuracy', 'log_loss'],
            ascending=[False, True]
        ).reset_index(drop=True)

        df_mean = df.ix[:, df.columns != 'split'] \
            .groupby(['classifier']) \
            .mean() \
            .sort_values(
                by=['accuracy', 'log_loss'],
                ascending=[False, True])

        return df_mean if mean else df


# def create_mlp():
#     model = Sequential()
#     model.add(Dense(64, 'uniform', 'sigmoid', input_dim=input_dim))
#     # model.add(ActivityRegularization(l1=0, l2=0.001))
#     model.add(Dropout(0.2))
#     model.add(Dense(output_dim=64, activation='tanh'))
#     model.add(Dropout(0.1))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer='adam',
#         metrics=['accuracy']
#     )
#     return model


classifiers = {
    # 'keras_mlp': KerasClassifier(
    #     build_fn=create_mlp,
    #     nb_epoch=150,
    #     batch_size=64
    # ),
    'svc_linear': LinearSVC(),
    'lr_lbfgs': LogisticRegression(
        C=2.02739770e+04,  # particle swarm optimised
        tol=6.65926091e-04,
        solver='lbfgs'
    ),
    'lr_lbfgs_pso': LogisticRegression(
        C=5.76005997e+02,  # particle swarm optimised
        tol=7.05315544e-04,
        solver='lbfgs'
    ),
    'lr_lbfgs_default': LogisticRegression(solver='lbfgs'),
    'pa': PassiveAggressiveClassifier(
        C=0.01,
        fit_intercept=True,
        loss='hinge'
    ),
    'pa_default': PassiveAggressiveClassifier(),
    'gnb': GaussianNB(),
    'lda': LinearDiscriminantAnalysis(),
    'rf': RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=4,
        min_samples_leaf=3,
        min_samples_split=3
    ),
    # 'xgb': XGBClassifier(
    #     n_estimators=200,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     gamma=1,
    #     objective='binary:logistic',
    #     nthread=-1
    # ),
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=input_dim)),
])


if __name__ == '__main__':
    # path = '/home/rokkuran/workspace/stegasawus'
    path = 'c:/workspace/stegasawus'
    # path_train = '{}/data/features/train_shuffle_iter.csv'.format(path)
    path_train = '{}/data/features/train.csv'.format(path)

    train = pd.read_csv(path_train)

    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y = le_target.transform(train[target])

    drop_cols = [target, 'filename', 'seq_method', 'img_msg_dim']
    train = train.drop(drop_cols, axis=1)
    X = train.as_matrix()

    splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    mc = ModelComparer(X, y, pipeline, splitter, classifiers)
    mc.model_comparison()
    print '\n', mc.scores()
