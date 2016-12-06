import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from scipy import stats
from pyswarm import pso

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
def gs_parameter_tuning(clf, X_train, y_train, parameters, scoring, cv=5):
    gs_clf = GridSearchCV(clf, parameters, scoring=scoring, cv=cv, n_jobs=6)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


# TODO: PSO - generalise function, investigate performance, try dim reduction.
# TODO: PSO - work out how to use integer values.
def pso_parameter_tuning(clf, X, y, lb, ub, swarmsize, maxiter, n_splits=3):
    """
    Particle swarm optimisation based parameter tuning.

    Parameters
    ----------
    clf : sklearn classifier or pipeline
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
    def minimise(x):
        C, tol = x

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('clf', LogisticRegression(C=C, tol=tol, solver='lbfgs', n_jobs=6))
            ('clf', LinearSVC(C=C, tol=tol))
        ])

        ss = ShuffleSplit(n_splits=n_splits, test_size=0.2)

        ll = []
        for i, (train_idx, val_idx) in enumerate(ss.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = pipeline.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            ll.append(metrics.log_loss(y_val, y_pred))
        return np.mean(ll)

    g, f = pso(
        minimise,
        lb,
        ub,
        swarmsize=swarmsize,
        maxiter=maxiter,
        debug=True
    )
    return g, f


# ******************************************************************************
if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    # path_train = '{}/data/features/train.csv'.format(path)
    path_train = '{}/data/features/train_lenna_identity.csv'.format(path)

    train = pd.read_csv(path_train)

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
    # train = train.drop([target, 'image'], axis=1)

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
    }

    # **************************************************************************
    parameters = yaml.safe_load(
        open('{}/stegasawus/parameter_tuning.yaml'.format(path), 'rb')
    )

    # Grid search parameter tuning.
    name = 'svc_linear'
    pipeline = Pipeline([
        ('features', combined_features),
        (name, classifiers[name]),
    ])

    gs_parameter_tuning(
        clf=pipeline,
        X_train=train.as_matrix(),
        y_train=y_train_binary,
        cv=3,
        parameters=parameters[name],
        scoring='accuracy'
    )

    # Particle swarm optimisation parameter tuning.
    lb = [1e-2, 1e-4]
    ub = [1e5, 1e-3]
    g, f = pso_parameter_tuning(
        clf=None,
        X=train.as_matrix(),
        y=y_train_binary,
        lb=lb,
        ub=ub,
        swarmsize=50,
        maxiter=10,
        n_splits=1
    )
    print g, f
