from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def create_cnn_small():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 256, 256)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model


path = '/home/rokkuran/workspace/stegosawus'
path_images = '{}/images/originals/'.format(path)
path_train = '{}/images/train/'.format(path)
path_validation = '{}/images/validation/'.format(path)
path_cropped = '{}/images/train/cropped/'.format(path)
path_output = '{}/images/train/encoded/'.format(path)
filepath_message = '{}/message.txt'.format(path)


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=path_train,  # this is the target directory
    batch_size=32,
    class_mode='binary'
)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    path_validation,
    batch_size=32,
    class_mode='binary'
)

model = create_cnn_small()
model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=20,
    validation_data=validation_generator,
    nb_val_samples=1000
)

model.save_weights('{}weights_initial.h5'.format(path_train))  # always save your weights after training or during training
