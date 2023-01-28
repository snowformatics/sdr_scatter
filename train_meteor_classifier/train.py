from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from imutils import paths
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.imagenet_utils import decode_predictions

# Train meteor CNN with rois extracted from "extract_meteor.py"


def train_cnn(model_name, resize_size, num_classes):
    """Tain CNN on meteor rois."""
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = list(paths.list_images("E:/sdr/meteors/classification/"))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize it to 130x100 pixels, and store the image in the
        # data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, resize_size)
        data.append(image)
        # extract the class label from the image path folder and update the
        # labels list
        if imagePath.find('area') != -1:
            label = 0
        elif imagePath.find('gerade_rechts') != -1:
            label = 1
        elif imagePath.find('gerade_unten') != -1:
            label = 2
        elif imagePath.find('point') != -1:
            label = 3
        elif imagePath.find('s') != -1:             # S shape
            print(imagePath)
            label = 4

        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    unique, counts = np.unique(trainY, return_counts=True)
    print(dict(zip(unique, counts)))

    y_train = np_utils.to_categorical(trainY)
    y_test = np_utils.to_categorical(testY)

    # # # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(100, 130, 3), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 100
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    H = model.fit(trainX, y_train, validation_data=(testX, y_test), nb_epoch=epochs, batch_size=32)
    # Final evaluation of the model
    scores = model.evaluate(testX, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (SmallVGGNet)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(model_name + '.png')
    model.save(model_name + '.h5')


def classify_object(model_name, path_roi, resize_size):
    """Predict  meteor class with CNN."""
    cnn_model = load_model(model_name)
    imagePaths = list(paths.list_images(path_roi))

    # Loop over all images to predict
    for i in imagePaths:
        roi_org = cv2.imread(i)
        try:
            roi_resized = cv2.resize(roi_org, resize_size)

            img_tensor = image.img_to_array(roi_resized)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            # Remember that the model was trained on inputs
            # that were preprocessed in the following way:
            img_tensor /= 255.
            preds = cnn_model.predict(img_tensor)

            predictions = 'area:' + str(int(round(preds[0][0] * 100, 0))) + ' trail:' + str(int(round(preds[0][1] * 100, 0))) + ' head:' + str(int(round(preds[0][2] * 100, 0))) + ' point:' + str(int(round(preds[0][3] * 100, 0)))

            print (i, predictions)
            cv2.imshow('image', roi_org)
            cv2.waitKey(0)

        except cv2.error:
            pass


train_cnn('my_model', (130, 100), 4)
classify_object('my_model', path, (130, 100))
