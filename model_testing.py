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
    path_train = '{}/data/train.csv'.format(path)

    train = pandas.read_csv(path_train)

    # target and index preprocessing
    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y_train_binary = le.transform(train[target])

    train = train.drop([target, 'image'], axis=1)

    combined_features = FeatureUnion([
        ('scaler', StandardScaler()),
    ])

    classifiers = {
        # 'classifier_knn': Pipeline([
        #     ('classifier_knn', KNeighborsClassifier()),
        # ]),
        'classifier_svc': Pipeline([
            ('classifier_svc', SVC(C=0.1)),
        ]),
        # 'classifier_lda': Pipeline([
        #     ('classifier_lda', LinearDiscriminantAnalysis()),
        # ]),
    }

    log_cols = ['classifier', 'acc', 'log_loss']
    log = pandas.DataFrame(columns=log_cols)

    # data splitting for cross validation
    sss = StratifiedShuffleSplit(
        y=y_train_binary,
        n_iter=1,
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
