import csv
import random
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import tensorflow.keras as keras

class PMIdata(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, targets):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.targets = targets

    def __len__(self):
        return len(self.targets) // self.batch_size

    def __getitem__(self, index):
        '''return tuple (input, target) for batch number index'''
        i = index * self.batch_size
        inputs = self.input_img_paths[i: i + self.batch_size]
        targets = self.targets[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + (self.img_size) + (3,), dtype = 'float32')
        for j, inp in enumerate(inputs):
            img = load_img(inp, target_size = self.img_size)
            x[j] = img
        return x, np.array(targets)

