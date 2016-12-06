import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from scipy import stats
from pyswarm import pso
from collections import Counter

from sklearn import metrics
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    LabelBinarizer,
    PolynomialFeatures)
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    ShuffleSplit)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, RFE

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def cv_split_generator(X, y, splitter):
    """
    Train and validation set split generator.

    Parameters
    ----------
    X : numpy.ndarray
        Feature data.
    y : numpy.ndarray
        Target data.
    splitter : sklearn splitter
        - ShuffleSplit
        - StratifiedShuffleSplit

    Yields
    ------
    i, X_train, X_val, y_train, y_val : data split

    i : int
        Iteration count.
    X_train : numpy.ndarray
        Feature training data.
    X_val : numpy.ndarray
        Feature validation data.
    y_train : numpy.ndarray
        Target training data.
    y_val : numpy.ndarray
        Target validation set.

    """
    for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield i, X_train, X_val, y_train, y_val


def scoring_metrics(y_true, y_pred, return_string=False):
    acc = metrics.accuracy_score(y_true, y_pred)
    ll = metrics.log_loss(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred)
    r = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)

    scores = [acc, ll, p, r, f1, roc_auc]

    s = 'acc = {:.2%}; ll = {:.4f}; p = {:.4f}; '.format(acc, ll, p)
    s += 'r = {:.4f}; f1 = {:.4f}; roc_auc = {:.4f}'.format(r, f1, roc_auc)

    if return_string:
        return scores, s
    else:
        return scores


def get_pipeline(name):
    combined_features = Pipeline([
        # ('features', FeatureUnion([
        #     ('scaler', StandardScaler()),
        #     # ('poly', PolynomialFeatures(
        #     #     degree=2,
        #     #     interaction_only=True,
        #     #     include_bias=False
        #     # ))
        # ])),
        ('pca', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=125)),
        ])),
        # ('kpca', KernelPCA(n_components=10)),
    ])
    pipeline = Pipeline([
        ('features', combined_features),
        (name, classifiers[name])
    ])
    return pipeline


def model_comparison(classifiers, X, y, splitter, cv_mean=True):
    scores = []
    score_cols = [
        'classifier', 'split', 'acc', 'log_loss',
        'precision', 'recall', 'f1', 'roc_auc']

    cv_splits = cv_split_generator(X=X, y=y, splitter=splitter)

    for i, X_train, X_val, y_train, y_val in cv_splits:
        for name, clf in classifiers.iteritems():
            pipeline = get_pipeline(name)
            estimator = pipeline.fit(X_train, y_train)

            y_pred = estimator.predict(X_val)
            m, ps = scoring_metrics(y_val, y_pred, return_string=True)
            ps += ' | {}_{}'.format(name, i)
            print ps
            scores.append([name, i] + m)

    scores = pd.DataFrame(scores, columns=score_cols)
    scores = scores.sort_values(
        by=['acc', 'log_loss'],
        ascending=[False, True]
    ).reset_index(drop=True)

    scores_mean = scores.ix[:, scores.columns != 'split'] \
        .groupby(['classifier']) \
        .mean() \
        .sort_values(by=['acc', 'log_loss'], ascending=[False, True]) \

    print '\n\n', scores_mean

    if cv_mean:
        return scores_mean
    else:
        return scores


def get_equal_sets(train):
    """Ensure approximate 50/50 cover stego split"""
    cols = train.columns
    a = train.as_matrix()
    np.random.shuffle(a)
    cover_half = a[a[:, 1] == 'cover']
    stego_half = a[a[:, 1] == 'stego'][:Counter(train.label)['cover']]
    train = pd.DataFrame(
        np.concatenate((cover_half, stego_half), axis=0),
        columns=cols
    )
    return train


classifiers = {
    'svc_linear': LinearSVC(),
    'lr_lbfgs': LogisticRegression(
        C=2.02739770e+04,  # particle swarm optimised
        tol=6.65926091e-04,
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
}


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    # path_train = '{}/data/features/train.csv'.format(path)
    path_train = '{}/data/features/train_lenna_identity.csv'.format(path)

    train = pd.read_csv(path_train)
    train = get_equal_sets(train)

    filenames = train.filename.copy()
    filenames = filenames.apply(
        lambda s: re.search(r'lenna\d+', s).group()
        if re.search(r'lenna\d+', s) is not None else 'cover'
    )

    # target and index preprocessing
    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y_train_binary = le_target.transform(train[target])

    train = train.drop([target, 'image', 'filename'], axis=1)

    cv_scores_mean = model_comparison(
        classifiers=classifiers,
        X=train.as_matrix(),
        y=y_train_binary,
        splitter=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
        cv_mean=True
    )
