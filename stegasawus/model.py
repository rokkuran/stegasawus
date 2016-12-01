import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from scipy import stats

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis)

from xgboost import XGBClassifier


# ******************************************************************************
def model_parameter_tuning(clf, X_train, y_train, parameters, scoring, cv=5):
    gs_clf = GridSearchCV(clf, parameters, scoring=scoring, cv=cv, n_jobs=6)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


# TODO: fixup and include particle swarm optimisation parameter tuning


def scoring_metrics(y_pred, y_true, return_string=False):
    acc = metrics.accuracy_score(y_true, y_pred)
    ll = metrics.log_loss(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred)
    r = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)

    scores = [acc, ll, p, r, f1, roc_auc]

    s = 'acc = {:.2%}; log loss = {:.4f}; p = {:.4f} '.format(acc, ll, p)
    s += 'r = {:.4f}; f1 = {:.4f}; roc_auc = {:.4f}'.format(r, f1, roc_auc)

    if return_string:
        return scores, s
    else:
        return scores


# ******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    # path_train = '{}/data/train_ac.csv'.format(path)
    # path_train = '{}/data/train_wavelet.csv'.format(path)
    path_train = '{}/data/train.csv'.format(path)

    train = pd.read_csv(path_train)

    # target and index preprocessing
    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y_train_binary = le_target.transform(train[target])

    train = train.drop([target, 'image'], axis=1)

    # **************************************************************************
    combined_features = Pipeline([
        ('features', FeatureUnion([
            ('scaler', StandardScaler()),
            # ('poly', PolynomialFeatures(
            #     degree=2,
            #     interaction_only=True,
            #     include_bias=False
            # ))
        ])),
        # ('pca', Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('pca', PCA()),
        # ])),
        # ('kpca', KernelPCA(n_components=10)),
    ])

    # **************************************************************************
    classifiers = {
        'knn': KNeighborsClassifier(
            n_neighbors=6,
            algorithm='ball_tree',
            weights='distance',
            metric='chebyshev'
        ),
        'knn_default': KNeighborsClassifier(),
        'svc_rbf': SVC(
            kernel='rbf',
            C=50,
            gamma=0.01,
            tol=1e-3
        ),
        'svc_rbf_default': SVC(kernel='rbf'),
        'svc_linear': LinearSVC(
            C=1e3,
            loss='squared_hinge',
            penalty='l2',
            tol=1e-3
        ),
        'svc_linear_default': LinearSVC(),
        'nusvc': NuSVC(),
        'rf': RandomForestClassifier(
            criterion='entropy',
            max_depth=12,
            min_samples_leaf=8,
            min_samples_split=5
        ),
        'rf_default': RandomForestClassifier(),
        'xgb': XGBClassifier(),
        'adaboost': AdaBoostClassifier(),
        'et': ExtraTreesClassifier(
            criterion='entropy',
            max_depth=25,
            min_samples_leaf=5,
            min_samples_split=5
        ),
        'et_default': ExtraTreesClassifier(),
        'gbc': GradientBoostingClassifier(),
        'lr_lbfgs': LogisticRegression(
            # C=1000,
            # tol=1e-3,
            C=3.23594105e+01,  # particle swarm optimised
            tol=6.83049831e-04,
            solver='lbfgs'
        ),
        'lr_lbfgs_default': LogisticRegression(),
        'pa': PassiveAggressiveClassifier(
            C=0.01,
            fit_intercept=True,
            loss='hinge'
        ),
        'pa_default': PassiveAggressiveClassifier(),
        'gnb': GaussianNB(),
        'lda': LinearDiscriminantAnalysis(),
        'qda': QuadraticDiscriminantAnalysis(),
    }

    # **************************************************************************
    parameters = yaml.safe_load(
        open('{}/stegasawus/parameter_tuning.yaml'.format(path), 'rb')
    )

    # **************************************************************************
    # name = 'svc_linear'
    # pipeline = Pipeline([
    #     ('features', combined_features),
    #     (name, classifiers[name]),
    # ])
    #
    # model_parameter_tuning(
    #     clf=pipeline,
    #     X_train=train.as_matrix(),
    #     y_train=y_train_binary,
    #     cv=3,
    #     parameters=parameters[name],
    #     scoring='accuracy'
    # )

    # **************************************************************************
    scores = []
    score_cols = [
        'classifier', 'split', 'acc', 'log_loss', 'precision', 'recall',
        'f1', 'roc_auc'
    ]

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )

    train = train.as_matrix()
    for i, (train_idx, val_idx) in enumerate(ss.split(train, y_train_binary)):
        X_train, X_val = train[train_idx], train[val_idx]
        y_train, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

        for name, clf in classifiers.iteritems():

            pipeline = Pipeline([
                ('features', combined_features),
                (name, clf)
            ])
            estimator = pipeline.fit(X_train, y_train)

            y_pred = estimator.predict(X_val)

            m, ps = scoring_metrics(y_val, y_pred, return_string=True)
            ps += ' | {}_{}'.format(name, i)
            print ps
            scores.append([name, i] + m)

    scores = pd.DataFrame(scores, columns=score_cols)
    scores = scores.sort_values(
        by=['acc', 'f1', 'roc_auc'],
        ascending=False
    ).reset_index(drop=True)

    scores_mean = scores.ix[:, scores.columns != 'split'] \
        .groupby(['classifier']) \
        .mean() \
        .sort_values(by=['acc', 'f1', 'roc_auc'], ascending=False) \

    print '\n\n', scores_mean
