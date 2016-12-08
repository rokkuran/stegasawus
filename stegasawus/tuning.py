from stegasawus.model import (
    cv_split_generator,
    get_pipeline,
    get_equal_sets)

import numpy as np
import pandas as pd
import yaml
import re
import collections
import functools

import matplotlib.pyplot as plt

from pyswarm import pso

from sklearn import metrics
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    PolynomialFeatures)
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    validation_curve,
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
def gs_parameter_tuning(clf, X_train, y_train, parameters, scoring, cv=5):
    gs_clf = GridSearchCV(clf, parameters, scoring=scoring, cv=cv, n_jobs=6)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


# TODO: improve, extend, refactor
def pso_parameter_tuning(clf, X, y, lb, ub, swarmsize, maxiter, n_splits=3,
                         integer=False, *args):
    """
    Particle swarm optimisation based parameter tuning.

    Parameters
    ----------
    clf : sklearn classifier
        Model to tune parameters.
    X : numpy.ndarray
        Training features.
    y : numpy.ndarray
        Training target values.
    lb : array_like
        Lower bound values for parameters to tune.
    ub : array_like
        Upper bound values for parameters to tune.
    swarmsize : int
        Number of particles in the swarm.
    maxiter : int
        Maximum number of iterations for swarm to search.
    n_splits : int, default = 3
        Number of cross validation splits.

    Returns
    -------
    g : array
        The swarm's best known parameters settings.
    f : scalar
        The value of the minimisation function at g.

    """
    def clf_check(clf, classifiers):
        return any([isinstance(clf, c) for c in classifiers])

    def minimise(x, *args):
        """"""
        if clf_check(clf, [LinearSVC, LogisticRegression]):
            C, tol = x
            clf.set_params(C=C, tol=tol)

        elif clf_check(clf, [RandomForestClassifier]):
            # random forest: all values need to be integer
            x = [int(np.round(v, 0)) for v in x]
            max_depth, min_samples_leaf, min_samples_split = x
            clf.set_params(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split)

        elif clf_check(clf, [XGBClassifier]):
            # xgb: max_depth should be integer
            max_depth, learning_rate, gamma = x
            max_depth = int(np.round(max_depth, 0))
            clf.set_params(
                max_depth=max_depth,
                learning_rate=learning_rate,
                gamma=gamma)

        else:
            raise Exception('Classifier not supported.')

        pipeline = Pipeline([
            ('pca', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=125)),
            ])),
            ('clf', clf)
        ])

        ss = ShuffleSplit(n_splits=n_splits, test_size=0.2)
        cv_splits = cv_split_generator(X=X, y=y, splitter=ss)

        ll = []
        for i, X_train, X_val, y_train, y_val in cv_splits:
            model = pipeline.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            ll.append(metrics.log_loss(y_val, y_pred))

        print x, np.mean(ll)
        return np.mean(ll)

    g, f = pso(minimise, lb, ub, swarmsize=swarmsize, maxiter=maxiter,
               debug=True, args=('clf', clf))
    return g, f


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
        criterion='gini',
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        min_samples_split=3
    ),
    'rf_default': RandomForestClassifier(),
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
    'qda': QuadraticDiscriminantAnalysis(),
    'xgb_defualt': XGBClassifier(),
    'xgb': XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=100,
        silent=True,
        objective='binary:logistic',
        nthread=-1,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        seed=0,
        missing=None
    )
}


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
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

    # **************************************************************************
    parameters = yaml.safe_load(
        open('{}/stegasawus/parameter_tuning.yaml'.format(path), 'rb')
    )

    # **************************************************************************
    # Grid search parameter tuning.
    def run_gs_parameter_tuning():
        name = 'knn'
        pipeline = get_pipeline(name)

        gs_parameter_tuning(
            clf=pipeline,
            X_train=train.as_matrix(),
            y_train=y_train_binary,
            cv=3,
            parameters=parameters['grid_search'][name],
            scoring='accuracy'
        )

    # run_gs_parameter_tuning()

    # **************************************************************************
    # Particle swarm optimisation parameter tuning.
    def run_pso_parameter_tuning(clf_name):

        # TODO: fix issue with string representations of '1e-3' in yaml read
        lb = [float(v) for v in parameters['pso'][clf_name]['lb']]
        ub = [float(v) for v in parameters['pso'][clf_name]['ub']]

        g, f = pso_parameter_tuning(
            clf=classifiers['lr_lbfgs'], X=train.as_matrix(), y=y_train_binary,
            lb=lb, ub=ub, swarmsize=100, maxiter=20, n_splits=3)
        print g, f

    run_pso_parameter_tuning('lr_lbfgs')
    # run_pso_parameter_tuning('rf')
    # run_pso_parameter_tuning('xgb')

    # **************************************************************************
    def plot_validation_curve():
        name = 'svc_linear'
        pipeline = get_pipeline(name)

        param_range = np.logspace(-2, 3, 6)
        # param_range = np.logspace(-5, -1, 5)
        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=train.as_matrix(),
            y=y_train_binary,
            param_name='%s__C' % name,
            # param_name='lr_lbfgs__tol',
            param_range=param_range,
            cv=5,
            scoring='accuracy',
            n_jobs=6
        )

        plt.semilogx(
            param_range, train_scores.mean(axis=1),
            ls='-', lw=1, color='b', alpha=1, label='train'
        )
        plt.fill_between(
            param_range,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            color='b', alpha=0.1, lw=0.5
        )
        plt.semilogx(
            param_range, val_scores.mean(axis=1),
            ls='-', lw=1, color='r', alpha=1, label='validation'
        )
        plt.fill_between(
            param_range,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1),
            color='r', alpha=0.1, lw=0.5
        )

        plt.title('%s: validation curve' % name)
        plt.xlabel('C')
        plt.ylabel('Score')
        plt.ylim(0.0, 1.1)
        plt.legend(loc="best")
        plt.show()

    # **************************************************************************
    def plot_roc_curve(name):
        pipeline = get_pipeline(name)

        ss = ShuffleSplit(n_splits=5, test_size=0.2)

        X = train.as_matrix()
        y = y_train_binary

        fpr, tpr = [], []
        for i, (train_idx, val_idx) in enumerate(ss.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = pipeline.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            fpr_i, tpr_i, _ = metrics.roc_curve(y_val, y_pred)
            fpr.append(fpr_i)
            tpr.append(tpr_i)

        fpr, tpr = np.array(fpr), np.array(tpr)

        plt.figure()
        plt.plot(
            fpr.mean(axis=0), tpr.mean(axis=0),
            color='b', alpha=0.6, lw=1, label='ROC curve'
        )
        plt.plot([0, 1], [0, 1], color='k', alpha=0.6, lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s: ROC Curve' % name)
        plt.legend(loc="lower right")
        plt.show()

    # plot_roc_curve('svc_linear')
