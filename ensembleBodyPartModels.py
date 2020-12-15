#!/usr/bin/env python
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os
import generator
import json
import tensorflow.keras as tfk
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

config = json.load(open('config.json'))


img_size = config['img_size']

df = pd.read_csv(config['data_file'], delimiter= ":", names = ['path','label','pmi'])
val_sample_ratio = config['val_sample_ratio']
batch_size = config['batch_size']
#df = pd.read_csv('head_train', delimiter= ":", names = ['name','label','pmi'])
classes = df['label'].unique()



def build_model(classname, train_gen, val_gen):
    model = Sequential()
    inputShape = (img_size, img_size, 3)

    model.add(Conv2D(32, (5, 5), padding="same",input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    
    lr = config['lr']
    if config['resume'] != 'true':
        lr = config['lr'] * 10
    optimizer = tensorflow.keras.optimizers.RMSprop(lr)
    model.compile(loss='mse',
        optimizer = optimizer,
        metrics=['mae', 'mse'])
    #checkpoint = ModelCheckpoint(classname + '.h5', save_best_only=True)
    callbacks = [tfk.callbacks.ModelCheckpoint(classname +".h5", save_best_only=True)]
    if config['resume'] == 'true':
        print("LOODING WEIGHTS FOR MODEL {}".format(classname +".h5"))
        nodel.load_weights(classname +".h5")

    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        callbacks=callbacks
    )
    return model


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
        # make predictions
        yhats = [model.predict(testX) for model in members]
        yhats = np.array(yhats)
        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        # argmax across classes
        result = np.argmax(summed, axis=1)
        return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
        # select a subset of members
        subset = members[:n_members]
        print(len(subset))
        # make prediction
        yhat = ensemble_predictions(subset, testX)
        # calculate accuracy
        return accuracy_score(testy, yhat)

def train_val(input_img_pahts, targets):
    samples_counts = len(input_img_paths)
    #shuffleing the inputs and the targets with the same seed (1345)
    random.Random(1345).shuffle(input_img_paths)
    random.Random(1245).shuffle(targets)

    train_index = int((1 - val_sample_ratio)*samples_counts)
    val_index = int(-1 * val_sample_ratio * samples_counts)
    train_inputs = input_img_paths[:train_index]
    train_targets = targets[:train_index]

    val_inputs = input_img_paths[val_index:]
    val_targets = targets[val_index:]
    return train_inputs, train_targets, val_inputs, val_targets

if __name__=='__main__':
    #models = []
    for classname in classes:
        df_sub = df[df['label'] == classname]
        input_img_paths = []
        targets = []
        for index, line in df_sub.iterrows():
            img_path = line['path']
            target = line['pmi']
            input_img_paths.append(img_path)
            targets.append(target)
        train_inputs, train_targets, val_inputs, val_targets = train_val(input_img_paths, targets)
        train_gen = generator.PMIdata(batch_size, (img_size, img_size), train_inputs, train_targets)
        val_gen = generator.PMIdata(batch_size, (img_size, img_size), val_inputs, val_targets)
        build_model(classname, train_gen, val_gen)
        #models.append(build_model(classname, train_gen, val_gen))
    '''
    scores = list()
    for n_model in range(1, len(classes) + 1):
        score = evaluate_n_members(models, n_model, testX, testy)
        print('> %.3f' % score)
        scores.append(score)

    # plot score vs number of ensemble members
    x_axis = [i for i in range(1, len(classes) + 1)]
    plt.plot(x_axis, scores)
    plt.savefig('ensemble.png')
    #pyplot.show()
    '''




