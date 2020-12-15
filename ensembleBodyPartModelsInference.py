#!/usr/bin/env python

from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras
import numpy as np
import csv
import json
import pandas as pd

config = json.load(open('config.json'))
df = pd.read_csv(config['data_file'], delimiter= ":", names = ['path','label','pmi'])

classes = df['label'].unique()
base_model_img_size = config['img_size']


def extract_days():
    df = pd.read_csv(config['all_pmis'], sep = ":", names=['path', 'pmi', 'month'])
    day_per_donor = {}
    for index, row in df.iterrows():
        donor = row['path'][:3]
        if donor not in day_per_donor:
            day_per_donor[donor] = df[df['path'].str.contains(donor)]['pmi'].sort_values().iloc[-1]
    return day_per_donor

day_per_donor = extract_days()

def small_net():
    # initialize the model
    model = Sequential()
    inputShape = (base_model_img_size, base_model_img_size, 3)
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
    model.add(Dense(1))
    optimizer = tensorflow.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def get_data(file_):
    not_found = 0
    X = []
    Y = []
    train_data_paths = file_

    with open(train_data_paths, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            try:
                pmi = int(row[2].strip())
                img = image.load_img(row[0].strip(),
                        target_size = (base_model_img_size,
                         base_model_img_size, 3), grayscale = False)
                img = image.img_to_array(img)
                img = img/255
                X.append(img)
                Y.append([pmi, day_per_donor[row[0].split('/')[-2]]])
            except:
                print("couldn't load", row)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y



# ## Predict with the small net
def sara_metric(y_true, y_pred):
    y_class = y_true[:,0]
    y_days = y_true[:,1]
    loss=(1/y_days)*(abs(y_class - y_pred))
    return loss

for class_ in classes:
    loaded_model = small_net()
    loaded_model.load_weights(class_ + '.h5')
    X, Y = get_data(config['test_data'])


    preds = loaded_model.predict(X)
    pred = preds[:,0].astype(int) #preds.argmax(axis=-1) #if classifier
        
    d = sara_metric(Y, pred)
    #print("d: {}".format(d))
    print("mean of prediction deviation using model {}: {} ".format(class_ + '.h5',np.mean(d)))

    ## TODO add voting


