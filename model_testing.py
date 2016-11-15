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
    # path_test = '{}/data/test.csv'.format(path)

    train = pandas.read_csv(path_train)
    # test = pandas.read_csv(path_test)

    # target and index preprocessing
    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    labels = le_target.transform(train[target])
    classes = list(le_target.classes_)

    image_names = train.pop('image')
    train = train.drop([target], axis=1)


    # scaler = StandardScaler().fit(train)
    # cols = train.columns
    # train = pandas.DataFrame(scaler.transform(train), columns=cols)
    #
    # cols = test.columns
    # test = pandas.DataFrame(scaler.transform(test), columns=cols)



    combined_features = FeatureUnion([
        ('univ', StandardScaler()),
        # ('polynomial_features', polynomial_features),
        # ('kpca_univ', KernelPCA(
        #     n_components=20,
        #     kernel="rbf",
        #     fit_inverse_transform=True,
        #     gamma=gamma
        # )),
        # # ('pca', pca),
        # ('poly_kpca', Pipeline([
        #     ('polynomial_features', polynomial_features),
        #     ('kpca_poly', kpca)
        # ])),
        # ('poly_pca', Pipeline([
        #     ('polynomial_features', polynomial_features),
        #     ('pca', PCA())
        # ])),
    ])


    classifiers = {
        # 'classifier_knn': Pipeline([
        #     ('classifier_knn', KNeighborsClassifier())
        # ]),
        # 'classifier_knn_gs': Pipeline([
        #     ('classifier_knn_gs', KNeighborsClassifier(
        #         n_neighbors=8,
        #         algorithm='ball_tree',
        #         weights='distance')
        #     )
        # ]),
        'classifier_lr_lbfgs': Pipeline([
            ('classifier_lr_lbfgs', LogisticRegression(
                C=7000,
                tol=1e-5,
                # multi_class= 'multinomial',
                penalty='l2',
                solver='lbfgs')
                # n_jobs=6)
            ),
        ]),
        'classifier_lr_ncg': Pipeline([
            ('classifier_lr_ncg', LogisticRegression(
                C=7000,
                tol=1e-5,
                # multi_class= 'multinomial',
                penalty='l2',
                solver='newton-cg')
                # n_jobs=6)
            ),
        ]),
        # 'classifier_lr': Pipeline([
        #     ('classifier_lr', LogisticRegression(penalty='l2', n_jobs=6)),
        # ]),
        # 'classifier_svc': Pipeline([
        #     ('classifier_svc', SVC(
        #         kernel='rbf',
        #         C=100,
        #         tol=1e-3)),
        #         # probability=True)),
        # ]),
        # 'classifier_svc_linear': Pipeline([
        #     ('classifier_svc_linear', SVC(
        #         kernel='linear',
        #         C=100,
        #         tol=1e-4)),
        #         # probability=True)),
        # ]),
        # 'classifier_svc_gs': Pipeline([
        #     ('classifier_svc_gs', SVC(kernel="linear", C=10, probability=True)),
        # ]),
        # 'classifier_nusvc': Pipeline([
        #     ('classifier_nusvc', NuSVC(probability=True)),
        # ]),
        'classifier_rf': Pipeline([
            ('classifier_rf', RandomForestClassifier(
                n_estimators=200,
                criterion='entropy',
                max_depth=10,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=5,
                n_jobs=6)
            ),
        ]),
        # 'classifier_xgb': Pipeline([
        #     ('classifier_xgb', XGBClassifier(
        #         n_estimators=200,
        #         objective="multi:softprob",
        #         max_depth=6,
        #         learning_rate=0.01,
        #         gamma=0,
        #         nthread=6)
        #     ),
        # ]),
        'classifier_gnb': GaussianNB(),
        # 'classifier_pa': PassiveAggressiveClassifier(),

        # 'classifier_lda': Pipeline([
        #     ('classifier_lda', LinearDiscriminantAnalysis()),
        # ]),
        # 'classifier_qda': Pipeline([
        #     ('classifier_lda', QuadraticDiscriminantAnalysis()),
        # ]),
    }

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pandas.DataFrame(columns=log_cols)

    # data splitting for cross validation
    sss = StratifiedShuffleSplit(
        y=labels,
        n_iter=1,
        test_size=0.2,
        random_state=0
    )

    estimators = dict()

    print("="*30)
    print('****Results****')
    for i, (train_index, test_index) in enumerate(sss):
        X_train, X_test = train.as_matrix()[train_index], train.as_matrix()[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        for name, clf in classifiers.iteritems():
            # print("="*30)
            # print(name)
            # pipeline = Pipeline([('features', combined_features), (name, clf)])
            # estimator = pipeline.fit(X_train, y_train)
            estimator = clf.fit(X_train, y_train)
            # name = clf.__class__.__name__
            estimators['{}_{}'.format(name, i)] = estimator

            # print('****Results****')
            train_predictions = estimator.predict(X_test)
            acc = metrics.accuracy_score(y_test, train_predictions)
            # print("Accuracy: {:.4%}".format(acc))

            train_predictions = estimator.predict(X_test)
            ll = metrics.log_loss(y_test, train_predictions)
            # print("Log Loss: {}".format(ll))

            print 'acc: {:.4%}; log loss: {:.4f} | {}_{}'.format(acc, ll, name, i)

            log_entry = pandas.DataFrame(
                [['{}_{}'.format(name, i), acc*100, ll]],
                columns=log_cols
            )
            log = log.append(log_entry)

    print("="*30)
