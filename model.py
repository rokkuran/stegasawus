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

import os
import skimage
import skimage.io as io


path = '/home/rokkuran/workspace/stegosawus'
path_images = '{}/images/originals/'.format(path)
path_cropped = '{}/images/cropped/'.format(path)
path_output = '{}/images/encoded/'.format(path)
filepath_message = '{}/message.txt'.format(path)
