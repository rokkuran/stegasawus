import numpy
import pandas

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    LabelBinarizer,
    PolynomialFeatures)
from sklearn import metrics, cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, learning_curve, ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis)

from xgboost import XGBClassifier


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    # path_train = '{}/data/train.csv'.format(path)
    path_train = '{}/data/train_catdog_ac.csv'.format(path)

    train = pandas.read_csv(path_train)

    # target and index preprocessing
    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y_train_binary = le_target.transform(train[target])

    train = train.drop([target, 'image'], axis=1)

    combined_features = FeatureUnion([
        ('scaler', StandardScaler()),
        # ('pca', KernelPCA(n_components=10)),
    ])

    classifiers = {
        'classifier_knn': Pipeline([
            ('classifier_knn', KNeighborsClassifier()),
        ]),
        'classifier_svc': Pipeline([
            # ('pca', PCA(n_components=20)),
            ('classifier_svc', SVC()),
        ]),
        'classifier_lda': Pipeline([
            ('classifier_lda', LinearDiscriminantAnalysis()),
        ]),
        'classifier_rf': Pipeline([
            ('classifier_rf', RandomForestClassifier()),
        ]),
        'classifier_lr': Pipeline([
            ('classifier_lr', LogisticRegression()),
        ]),
        'classifier_rf': Pipeline([
            ('classifier_gnb', GaussianNB()),
        ]),
    }

    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import classification_report
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    # train.as_matrix(), y_train_binary, test_size=0.2, random_state=0)
    #
    # tuned_parameters = [
    #     # {
    #     #     'kernel': ['rbf'],
    #     #     'gamma': [1e-3, 1e-4],
    #     #     # 'C': [1, 10, 100, 1000]
    #     #     'C': [1, 10, 100, 1000]
    #     # },
    #     {
    #         # 'kernel': ['linear'],
    #         # 'C': [1, 10, 100, 1000]
    #     },
    #     {
    #         'svc__kernel': ['rbf'],
    #         # 'svc__gamma': [1e-3, 1e-4],
    #         # 'C': [1, 10, 100, 1000]
    #         'svc__C': [0.01, 0.1, 1, 10, 100, 1000]
    #     },
    # ]
    #
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print
    #
    #     clf = GridSearchCV(
    #         Pipeline([
    #             ('scaler', StandardScaler()),
    #             # ('pca', KernelPCA(n_components=20)),
    #             ('svc', SVC(C=1, kernel='rbf')),
    #         ]),
    #         tuned_parameters, cv=5, scoring='%s_macro' % score
    #     )
    #     clf.fit(X_train, y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print(clf.best_params_)
    #     print
    #     print("Grid scores on development set:")
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print
    #
    #     print("Detailed classification report:")
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print
    #     y_true, y_pred = y_test, clf.predict(X_test)
    #     print(classification_report(y_true, y_pred))
    #     print

    log_cols = ['classifier', 'acc', 'log_loss']
    log = pandas.DataFrame(columns=log_cols)

    # data splitting for cross validation
    sss = StratifiedShuffleSplit(
        y=y_train_binary,
        n_iter=5,
        test_size=0.2,
        random_state=0
    )

    estimators = dict()

    print("="*48)
    print('*********************Results*********************')
    for i, (train_index, val_index) in enumerate(sss):
        X_train, X_val = train.as_matrix()[train_index], train.as_matrix()[val_index]
        y_train, y_val = y_train_binary[train_index], y_train_binary[val_index]

        for name, clf in classifiers.iteritems():
            pipeline = Pipeline([('features', combined_features), (name, clf)])
            estimator = pipeline.fit(X_train, y_train)
            # estimator = clf.fit(X_train, y_train)
            estimators['{}_{}'.format(name, i)] = estimator

            train_predictions = estimator.predict(X_val)
            acc = metrics.accuracy_score(y_val, train_predictions)

            train_predictions = estimator.predict(X_val)
            ll = metrics.log_loss(y_val, train_predictions)

            print 'acc: {:.4%}; log loss: {:.4f} | {}_{}'.format(acc, ll, name, i)

            log_entry = pandas.DataFrame(
                [['{}_{}'.format(name, i), acc*100, ll]],
                columns=log_cols
            )
            log = log.append(log_entry)

    print("="*48)
