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


df = pd.read_csv('PMIs', sep = ":", names=['path', 'pmi'])
donors = {}
for index, row in df.iterrows():
    if row['path'].split('/')[-2] not in donors:
        donors[row['path'].split('/')[-2]] = df[df['path'].str.contains(row['path'].split('/')[-2])]['pmi'].sort_values().iloc[-1]


def my_loss(y_true, y_pred):
    y_class = y_true[:,:-1]
    y_days = y_true[:,-1]
    loss=(1/y_days)*(tensorflow.keras.losses.categorical_crossentropy(y_class, y_pred))
    print(loss)
    return loss

def get_data(path):
    not_found = 0
    X = []
    Y = []
    Days = []
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
                Days.append(donors[row[0].split('/')[-2]])
            except:
                not_found += 1

    X = np.array(X)
    Y = np.array(Y)
    try:
        print("X.shape {}, Y.shape {}, not_found {}".format(X.shape, Y.shape, not_found))
    except:
        pass
    return X, Y, Days

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
    #test_preds = []
    #for p in preds:
    #    test_preds.append(p[0])
    test_preds = preds.argmax(axis=-1)
    get_metrics(test_preds, test_labels.argmax(axis=-1))
    ##### val data
    print("Val data:")
    preds = model.predict(val_imgs)

    #val_preds = []
    #for p in preds:
    #    val_preds.append(p[0])
    val_preds = preds.argmax(axis=-1)
    get_metrics(val_preds, val_labels.argmax(axis=-1))
    import bpython
    bpython.embed(locals())

def small_net(width, height, depth, train_imgs, train_labels, val_imgs, val_labels):
    num_classes = train_labels.shape[1] - 1 # for the day value
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
    model.add(Dropout(0.5))
    # linear regression
    ##model.add(Dense(1))
    ##model.add(Activation("linear"))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)
    #es = EarlyStopping(monitor='val_mean_squared_error', mode='auto', verbose=1, patience=100, 
    #restore_best_weights=True)

    model.compile(optimizer = sgd, loss = my_loss, metrics=['accuracy'])
    #model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    #checkpoint = ModelCheckpoint(str(fine_tune)+str(rand_init)+ str(training) + network + 
    #'epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', 
    checkpoint = ModelCheckpoint('January_epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='max')

    #model.load_weights('foot_models/foot_epoch_-280-_loss_0.852961-_val_loss_5.98759.h5')
    model.load_weights('Best2_January_epoch_-008-_loss_1.598702-_val_loss_8.77076.h5')
    print(model.layers[0].get_weights())
    model.summary()
    model.pop()
    model.add(Dense(num_classes, activation='softmax'))
    print(model.layers[0].get_weights())
    model.summary()

    batch_size = 256
    history = model.fit(
        train_imgs,
        train_labels,
        batch_size=batch_size,
        epochs=500,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_imgs, val_labels),
        callbacks=[checkpoint]#, es]
    )
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    # return the constructed network architecture
    import bpython
    bpython.embed(locals())
    return model

def add_days(Days, Y):
    Y_new = []
    for index, label in enumerate(Y):
        label = list(label)
        label.append(Days[index])
        Y_new.append(label)
    return Y_new

## Read the train data
X, Y, Days= get_data(train_data_paths)
print("X.shape {}, Y.shape {}".format(X.shape, Y.shape))
num_classes = Y.max() + 1
Y = tensorflow.keras.utils.to_categorical(Y, num_classes=num_classes)
#Y_new = add_days(Days, Y)
#Y = np.array(Y_new)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(X, Y,test_size=0.3, random_state=42)
test_imgs, test_labels, Days = get_data(train_data_paths+'_test')
test_labels = tensorflow.keras.utils.to_categorical(test_labels , num_classes=num_classes)
#test_labels_new = add_days(Days, test_labels)
#test_labels = np.array(test_labels_new)

## Run the model
networks = ['resnet']#, 'vgg']
rand_init = [False]#[True, False]
training = [False]#, True,False]
fine_tune = [True]#[True, False]
for net in networks:
    for init in rand_init:
        for mode in training:
            for train_type in fine_tune:
                if init == True and mode == True:
                    continue
                else:
                    print("network: {},rand init: {}, training: {}, fine_tune:{}".
                            format(net, init, mode, train_type))
                    #model = create_model(train_imgs, train_labels, val_imgs, 
                    #val_labels, fine_tune=train_type, rand_init=init, training=mode, network=net)
                    model = small_net(base_model_img_size, base_model_img_size, 3, 
                            train_imgs, train_labels, val_imgs, val_labels)
                    cnn_eval(model, test_imgs, test_labels, val_imgs, val_labels)
                    #random_forest_regresor(inp, model, train_imgs, train_labels,val_imgs, 
                    #val_labels, test_imgs, test_labels, fine_tune=train_type, 
                    #rand_init=init, training=mode, network=net)
                
