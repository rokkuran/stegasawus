import numpy
import pandas

# import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    LabelBinarizer,
    PolynomialFeatures)
from sklearn import metrics
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    ShuffleSplit,
    StratifiedShuffleSplit)
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
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis)

from xgboost import XGBClassifier


#*******************************************************************************
path = '/home/rokkuran/workspace/stegasawus'
path_train = '{}/data/train.csv'.format(path)

train = pandas.read_csv(path_train)

# target and index preprocessing
target = 'label'
le_target = LabelEncoder().fit(train[target])
y_train_binary = le_target.transform(train[target])

train = train.drop([target, 'image'], axis=1)

#*******************************************************************************
combined_features = FeatureUnion([
    ('scaler', StandardScaler()),
    # ('pca', KernelPCA(n_components=10)),
])

#*******************************************************************************
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
    'svc_linear': SVC(
        kernel='linear',

    ),
    'svc_linear_default': SVC(kernel='linear'),
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
        C=1000,
        tol=1e-3,
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

#*******************************************************************************
def model_parameter_tuning(clf, X_train, y_train, parameters, scoring, cv=5):
    gs_clf = GridSearchCV(clf, parameters, scoring=scoring, cv=cv, n_jobs=6)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


parameters = {
    'svc_rbf': {
        'svc_rbf__C': [1, 50, 100, 250, 600, 650, 750, 800, 900, 1000],
        'svc_rbf__tol': [1e-3, 1e-4],
        'svc_rbf__gamma': [0.01, 0.1, 0.25, 0.5, 0.75],
    },
    'svc_linear': {
        'svc_linear__C': [1, 50, 100, 250, 600, 650, 750, 800, 900, 1000],
        'svc_linear__tol': [1e-3, 1e-4],
        'svc_linear__gamma': [0.01, 0.1, 0.25, 0.5, 0.75],
    },
    'nusvc': {
        'nusvc__nu': [0.01, 0.1, 0.25, 0.5, 0.75, 0.90],
        'nusvc__kernel': ['rbf', 'poly', 'sigmoid'],
        'nusvc__tol': [1e-3, 1e-4],
        'nusvc__gamma': [0.01, 0.1, 0.25, 0.5, 0.75],
    },
    'knn': {
        'knn__n_neighbors': [3, 6, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'knn__metric': ['minkowski', 'euclidean', 'chebyshev', 'manhattan'],
    },
    'bag_knn': {
        'bag_knn__base_estimator__n_neighbors': [3, 6, 9],
        'bag_knn__max_samples': [0.1, 0.25, 0.5, 0.75, 0.90],
        'bag_knn__max_features': [0.1, 0.25, 0.5, 0.75, 0.90],
    },
    'lr_lbfgs': {
        'lr_lbfgs__C': numpy.logspace(-3, 4, 8),
        'lr_lbfgs__tol': [1e-3, 1e-4]
    },
    'rf': {
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [10, 12, 15, 20],
        'rf__min_samples_leaf': [1, 5, 8, 12],
        'rf__min_samples_split': [2, 3, 4, 5],
    },
    'et': {
        'et__criterion': ['gini', 'entropy'],
        'et__max_depth': [6, 10, 18, 20, 25],
        'et__min_samples_leaf': [1, 5, 8, 12],
        'et__min_samples_split': [2, 3, 5, 8, 10],
    },
    'pa': {
        'pa__C': numpy.logspace(-3, 4, 8),
        'pa__fit_intercept': [False, True],
        'pa__loss': ['hinge', 'squared_hinge'],
    }
}

#*******************************************************************************
name = 'svc_linear'
pipeline = Pipeline([
    ('features', combined_features),
    (name, classifiers[name]),
])

model_parameter_tuning(
    clf=pipeline,
    X_train=train.as_matrix(),
    y_train=y_train_binary,
    cv=5,
    parameters=parameters[name],
    scoring='accuracy'
)

#*******************************************************************************
score_cols = [
    'classifier', 'split', 'acc', 'log_loss', 'precision', 'recall',
    'f1_score', 'roc_auc'
]

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=0
)

for i, (train_index, val_index) in enumerate(sss.split(train, y_train_binary)):
    X_train, X_val = train.as_matrix()[train_index], train.as_matrix()[val_index]
    y_train, y_val = y_train_binary[train_index], y_train_binary[val_index]

    for name, clf in classifiers.iteritems():
        pipeline = Pipeline([('features', combined_features), (name, clf)])
        estimator = pipeline.fit(X_train, y_train)

        train_predictions = estimator.predict(X_val)

        acc = metrics.accuracy_score(y_val, train_predictions)
        ll = metrics.log_loss(y_val, train_predictions)
        p = metrics.precision_score(y_val, train_predictions)
        r = metrics.recall_score(y_val, train_predictions)
        f1 = metrics.f1_score(y_val, train_predictions)
        roc_auc = metrics.roc_auc_score(y_val, train_predictions)

        s = 'acc = {:.2%}; log loss = {:.4f}; p = {:.4f} '.format(acc, ll, p)
        s += 'r = {:.4f}; f1 = {:.4f}; roc_auc = {:.4f}'.format(r, f1, roc_auc)
        s += ' | {}_{}'.format(name, i)
        print s

        scores.append([name, i, acc*100, ll, p, r, f1, roc_auc])

clf_acc = pandas.DataFrame(scores, columns=score_cols)
print log.sort_values(by=['acc'], ascending=[False])
