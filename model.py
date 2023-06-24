import numpy as np
import cv2
import os
import glob
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dropout, Dense, MaxPooling2D, Flatten
from tensorflow.python.keras.utils import layer_utils

# INITIAL PARAMETERS
epochs = 100
lr = 1e-3
batch_size = 64
image_dimension = (96, 96, 3)

data = []
labels = []

# LOADING IMAGES FROM THE DATASET
images = [image for image in glob.glob(r"<FILE PATH>\dataset" + "//**/*", recursive=True) if not os.path.isdir(image)]
random.shuffle(images)


# DATASET PRE-PROCESSING

# CONVERTING IMAGES TO ARRAYS AND LABELLING THE GENDER
for image in images:
    img = cv2.imread(image)

    img = cv2.resize(img, (image_dimension[0], image_dimension[1]))
    img = img_to_array(img)
    data.append(img)

    label = image.split(os.path.sep)[-2]

    if label == 'Female':
        label = 1
    else:
        label = 0
    
    labels.append([label])

# CONVERTING THEM INTO NUMPY ARRAYS
data = np.array(data, dtype='float') / 255
labels = np.array(labels)

# SPLITTING THE DATASET FOR TRAINING AND TESTING
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# AUGMENTING THE DATA
augment = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# BUILDING MODEL
def build_model(width, height, depth, classes):
    layers = Sequential()
    input = (height, width, depth)
    channel_dimension = -1

    if tf.keras.backend.image_data_format() == 'channels_first':
        input = (depth, height, width)
        channel_dimension = 1
    
    layers.add(Conv2D(32, (3,3), padding='same', input_shape=input))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization(axis=channel_dimension))
    layers.add(MaxPooling2D(pool_size=(3,3)))
    layers.add(Dropout(0.25))

    layers.add(Conv2D(64, (3,3), padding="same"))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization(axis=channel_dimension))

    layers.add(Conv2D(64, (3,3), padding="same"))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization(axis=channel_dimension))
    layers.add(MaxPooling2D(pool_size=(2,2)))
    layers.add(Dropout(0.25))

    layers.add(Conv2D(128, (3,3), padding="same"))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization(axis=channel_dimension))

    layers.add(Conv2D(128, (3,3), padding="same"))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization(axis=channel_dimension))
    layers.add(MaxPooling2D(pool_size=(2,2)))
    layers.add(Dropout(0.25))

    layers.add(Flatten())
    layers.add(Dense(1024))
    layers.add(Activation("relu"))
    layers.add(BatchNormalization())
    layers.add(Dropout(0.5))

    layers.add(Dense(classes))
    layers.add(Activation("sigmoid"))

    return layers


model = build_model(width=image_dimension[0], height=image_dimension[1], depth=image_dimension[2], classes=2)

# CHOOSING OPTIMIZER FOR THE MODEL
adam_optimizer = tf.keras.optimizers.Adam(lr=lr, decay=lr/epochs)

# COMPILING THE MODEL
model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

# TRAINING THE MODEL
gender_detector = model.fit_generator(augment.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // batch_size, epochs=epochs, verbose=1)

# SAVING THE TRAINED MODEL
model.save('trained_model.h5')