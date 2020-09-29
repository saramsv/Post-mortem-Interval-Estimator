from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow.keras
import numpy as np
np.random.seed(42) # for reproducibility
import csv
import pickle
import argparse
import sklearn.metrics, math
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='img2PMI_shuf_train')
args = parser.parse_args()

train_data_paths = args.data


base_model_img_size = 224
inp = tensorflow.keras.layers.Input((base_model_img_size , base_model_img_size , 3))


def get_data(path):
    not_found = 0
    X = []
    Y = []
    #Days = []
    train_data_paths = path
    with open(train_data_paths, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            try:
                pmi = int(row[1].strip())
                img = image.load_img(row[0].strip(),
                        target_size = (base_model_img_size,
                         base_model_img_size, 3), grayscale = False)

                img = image.img_to_array(img)
                img = img/255
                X.append(img)
                Y.append(pmi)
                #Days.append(donors[row[0].split('/')[-2]])
            except:
                not_found += 1

    X = np.array(X)
    Y = np.array(Y)
    try:
        print("X.shape {}, Y.shape {}, not_found {}".format(X.shape, Y.shape, not_found))
    except:
        pass
    return X, Y#, Days

def get_metrics(pred_labels, gt_labels):
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(gt_labels,pred_labels))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(gt_labels,pred_labels))
    print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(gt_labels,pred_labels)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(gt_labels,pred_labels))

def cnn_eval(model, test_imgs, test_labels, val_imgs, val_labels):
    score, acc = model.evaluate(test_imgs, test_labels)
    print("model eval: score {}, acc {}".format(score, acc))
    ##### test data
    print("Test data:")
    preds = model.predict(test_imgs)
    test_preds = preds[:,0].astype(int)
    get_metrics(test_preds, test_labels.argmax(axis=-1))
    ##### val data
    print("Val data:")
    preds = model.predict(val_imgs)
    val_preds = preds[:,0].astype(int)
    get_metrics(val_preds, val_labels.argmax(axis=-1))
    import bpython
    bpython.embed(locals())

def small_net(width, height, depth, train_imgs, train_labels, val_imgs, val_labels):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(32, (5, 5), padding="same",
            input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # second set of FC => RELU layers
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    # linear regression
    ##model.add(Dense(1))
    ##model.add(Activation("linear"))
    model.add(Dense(1))
    
    optimizer = tensorflow.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    checkpoint = ModelCheckpoint('June_regression_epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

    model.load_weights('June_regression_epoch_-017-_loss_199.297130-_val_loss_683.70539.h5')
    model.summary()
    model.pop()
    model.add(Dense(1))
    model.summary()

    batch_size = 256
    history = model.fit(
        train_imgs,
        train_labels,
        batch_size=batch_size,
        epochs=1,
        validation_data=(val_imgs, val_labels),
        callbacks=[checkpoint]#, es]
    )
    try:
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.clf()

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc.png')

    except:
        print("coundn't plot")
    import bpython
    bpython.embed(locals())
    # return the constructed network architecture
    return model

def add_days(Days, Y):
    Y_new = []
    for index, label in enumerate(Y):
        label = list(label)
        label.append(Days[index])
        Y_new.append(label)
    return Y_new

## Read the train data
X, Y= get_data(train_data_paths)
print("X.shape {}, Y.shape {}".format(X.shape, Y.shape))
train_imgs, val_imgs, train_labels, val_labels = train_test_split(X, Y,test_size=0.3, random_state=42)
test_imgs, test_labels = get_data(train_data_paths+'_test')

## Run the model
model = small_net(base_model_img_size, base_model_img_size, 3, 
        train_imgs, train_labels, val_imgs, val_labels)
cnn_eval(model, test_imgs, test_labels, val_imgs, val_labels)
