import os
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
import numpy as np
import pandas as pd
import json
import utilities


# image reshape from (160, 320, 3) to (100, 320, 3) after image cropping
IMAGE_SHAPE = (100, 320, 3)
DATA_PATH = './data/'


def save_model(model, model_filename, weights_filename):
    if Path(model_filename).is_file():
        os.remove(model_filename)

    with open(model_filename, 'w') as f:
        json.dump(model.to_json(), f)

    if Path(weights_filename).is_file():
        os.remove(weights_filename)

    model.save_weights(weights_filename)


def get_nvidia_model():
    model = Sequential()
    # normalize image input
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=IMAGE_SHAPE))
    # 1st CNN (100, 320, 3) to (48, 158, 24)
    model.add(Convolution2D(24, 5, 5, init='he_normal', subsample=(2, 2), name='conv1_1'))
    model.add(Activation('relu'))
    # 2nd CNN (48, 158, 24) to (22, 77, 36)
    model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    # 3rd CNN (22, 77, 36) to (9, 37, 48)
    model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    # 4th CNN (9, 37, 48) to (7, 35, 64)
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    # 5th CNN (7, 35, 64) to (5, 33, 64)
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    # flatten (5, 33, 64) to 1x10560
    model.add(Flatten())
    # 1st Fully-Connected 1x10560 to 1x1164
    model.add(Dense(1164, init='he_normal', name="dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 2nd Fully-Connected 1x1164 to 1x100
    model.add(Dense(100, init='he_normal', name="dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 3rd Fully-Connected 1x100 to 1x50
    model.add(Dense(50, init='he_normal', name="dense_2"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 4th Fully-Connected 1x50 to 1x10
    model.add(Dense(10, init='he_normal', name="dense_3"))
    model.add(Activation('relu'))
    # 5th Fully-Connected 1x10 to 1x1
    model.add(Dense(1, init='he_normal', name="dense_4"))
    # Adam optimizer to by mini square error
    model.compile(optimizer='adam', loss='mse')

    return model


def train_model(model, num_images):
    df = utilities.augment_dataframe(pd.read_csv(DATA_PATH + 'driving_log.csv'))

    df_sample = df.sample(num_images)
    train_features = []
    train_labels = []

    for _, row in df_sample.iterrows():
        image_path = DATA_PATH + row.image.strip()
        train_features.append(utilities.load_image(image_path, row.is_flipped))
        train_labels.append(row.steering)

    history = model.fit(np.array(train_features), np.array(train_labels), batch_size=10, nb_epoch=10, validation_split=0.2)
    save_model(model, 'model.json', 'model.h5')

    return history


model = get_nvidia_model()
num_images = 2000
train_model(model, num_images)