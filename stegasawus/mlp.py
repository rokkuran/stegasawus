from stegasawus.model import get_equal_sets, cv_split_generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, ActivityRegularization
from keras.regularizers import WeightRegularizer
from keras.wrappers.scikit_learn import KerasClassifier


input_dim = 125


def create_mlp():
    model = Sequential()
    model.add(Dense(64, 'uniform', 'sigmoid', input_dim=input_dim))
    # model.add(ActivityRegularization(l1=0, l2=0.001))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def plot_training_history(hist):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    ax1.plot(hist['acc'], '-', color='k', alpha=0.6, lw=1, label='acc')
    ax1.plot(hist['val_acc'], '-', color='r', alpha=0.6, lw=1, label='val_acc')
    ax1.set_xlabel('n_iterations')
    ax1.set_ylabel('accuracy', color='k')
    ax1.legend(loc='lower right')

    ax2.plot(hist['loss'], '-', color='purple', alpha=0.6, lw=1, label='loss')
    ax2.plot(hist['val_loss'], '-', color='b', alpha=0.6, lw=1, label='val_loss')
    ax2.set_xlabel('n_iterations')
    ax2.set_ylabel('loss', color='k')
    ax2.legend(loc='upper right')

    plt.savefig('{}/output/keras_mlp_training.png'.format(path))
    plt.show()


if __name__ == '__main__':
    path = '/home/rokkuran/workspace/stegasawus'
    path_train = '{}/data/features/train_lenna_identity.csv'.format(path)

    train = pd.read_csv(path_train)
    train = get_equal_sets(train)

    target = 'label'
    le_target = LabelEncoder().fit(train[target])
    y = le_target.transform(train[target])

    train = train.drop([target, 'image', 'filename'], axis=1)

    combined_features = Pipeline([
        ('pca', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=input_dim)),
        ])),
    ])

    X = combined_features.fit_transform(train.as_matrix())

    model = KerasClassifier(build_fn=create_mlp)

    splitter = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    cv_splits = cv_split_generator(X=X, y=y, splitter=splitter)

    scores = []
    hist = {}
    for i, X_train, X_val, y_train, y_val in cv_splits:
        X = combined_features.fit_transform(train.as_matrix())
        results = model.fit(
            X_train,
            y_train,
            nb_epoch=250,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )

        y_pred = model.predict(X_val)
        acc = metrics.accuracy_score(y_val, y_pred.flatten())
        scores.append(acc)

        hist[i] = results.history

    scores = np.array(scores)

    plot_training_history(hist[0])
