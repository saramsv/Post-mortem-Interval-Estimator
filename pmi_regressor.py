from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import * 
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
np.random.seed(42) # for reproducibility
import csv
import argparse
import sklearn.metrics, math
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='img2PMI_shuf_train')
args = parser.parse_args()

train_data_paths = args.data


base_model_img_size = 224
inp = keras.layers.Input((base_model_img_size , base_model_img_size , 3))


def get_data(path):
    not_found = 0
    X = []
    Y = []
    months= []
    train_data_paths = path
    with open(train_data_paths, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            try:
                pmi = int(row[1].strip())
                month = int(row[2].strip())
                img = image.load_img(row[0].strip(),
                        target_size = (base_model_img_size,
                         base_model_img_size, 3), grayscale = False)

                img = image.img_to_array(img)
                img = img/255
                X.append(img)
                Y.append(pmi)
                months.append(month)
            except:
                not_found += 1

    X = np.array(X)
    Y = np.array(Y)
    months = np.array(months)
    try:
        print("X.shape {}, Y.shape {}, not_found {}".format(X.shape,
            Y.shape, not_found))
    except:
        pass
    return X, Y, months

def get_metrics(pred_labels, gt_labels):
    print("Mean absolute error (MAE):      %f"
            % sklearn.metrics.mean_absolute_error(gt_labels,pred_labels))
    print("Mean squared error (MSE):       %f"
            % sklearn.metrics.mean_squared_error(gt_labels,pred_labels))
    print("Root mean squared error (RMSE): %f" 
            % math.sqrt(sklearn.metrics.mean_squared_error(gt_labels,pred_labels)))
    print("R square (R^2):                 %f"
            % sklearn.metrics.r2_score(gt_labels,pred_labels))

def cnn_eval(model, test_imgs, test_labels, test_month, val_imgs, val_labels, val_month):
    print("model eval: ",model.evaluate([test_imgs, test_month], test_labels))
    ##### test data
    print("Test data:")
    preds = model.predict([test_imgs, test_month])
    test_preds = preds[:,0].astype(int)
    get_metrics(test_preds, test_labels)#.argmax(axis=-1))
    ##### val data
    print("Val data:")
    preds = model.predict([val_imgs, val_month])
    val_preds = preds[:,0].astype(int)
    get_metrics(val_preds, val_labels)#.argmax(axis=-1))
    import bpython
    bpython.embed(locals())

def small_net(width, height, depth, train_imgs, train_labels, val_imgs, val_labels, train_month, val_month):
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
    
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    checkpoint = ModelCheckpoint('8donors_plus_month_epoch_-{epoch:03d}-_loss_{loss:03f}_val_loss_{val_loss:.5f}.h5', verbose=1, monitor='val_loss',save_best_only=True,\
            mode='min')

    model.load_weights('models/June_regression_epoch_-017-_loss_199.297130-_val_loss_683.70539.h5')
    model.pop()

    month_model = Sequential()
    month_model.add(Dense(1,  input_shape=(1,), activation='relu'))


    merged = concatenate([model.output, month_model.output])
    merged = Dense(1)(merged)

    from keras.models import Model
    new_model = Model([model.input, month_model.input], merged)

    new_model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    batch_size = 256
    history = new_model.fit(
        [train_imgs,train_month],
        train_labels,
        batch_size=batch_size,
        epochs=500,
        validation_data=([val_imgs, val_month], val_labels),
        callbacks=[checkpoint]
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
    return new_model

def add_days(Days, Y):
    Y_new = []
    for index, label in enumerate(Y):
        label = list(label)
        label.append(Days[index])
        Y_new.append(label)
    return Y_new

## Read the train data
X, Y, M= get_data(train_data_paths)
print("X.shape {}, Y.shape {}".format(X.shape, Y.shape))
train_imgs, val_imgs, train_labels, val_labels, train_month, val_month = train_test_split(X, Y, M, test_size=0.3, random_state=42)
test_imgs, test_labels, test_month = get_data(train_data_paths+'_test')

## Run the model
model = small_net(base_model_img_size, base_model_img_size, 3, 
        train_imgs, train_labels, val_imgs, val_labels, train_month, val_month)
cnn_eval(model, test_imgs, test_labels, test_month, val_imgs, val_labels, val_month)
