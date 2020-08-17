from keras.applications import ResNet50
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
import csv
import pickle
import argparse
import sklearn.metrics, math

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
    train_data_paths = path
    with open(train_data_paths, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            pmi = int(row[1].strip())
            img = image.load_img(row[0].strip(),
                    target_size = (base_model_img_size,
                     base_model_img_size, 3), grayscale = False)

            img = image.img_to_array(img)
            img = img/255
            X.append(img)
            Y.append(pmi)

    X = np.array(X)
    Y = np.array(Y)
    try:
        print("X.shape {}, Y.shape {}, not_found {}".format(X.shape, Y.shape, not_found))
    except:
        pass
    return X, Y

def get_metrics(pred_labels, gt_labels):
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(gt_labels,pred_labels))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(gt_labels,pred_labels))
    print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(gt_labels,pred_labels)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(gt_labels,pred_labels))

def cnn_eval(model, test_imgs, test_labels, val_imgs, val_labels):
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

    model2 = Model(inputs=model.input, outputs=model.layers[-3].output)
    test_features = []
    for i in range(len(test_imgs)):
        test_features.append(model2.predict(test_imgs[i:i+1]).flatten())
    test_features = np.array(test_features)
    print("test_features.shape:{}".format(test_features.shape))
    np.savetxt('test_embeddings_small_net_5000.tsv', 
            test_features, delimiter='\t')
    np.savetxt('test_labels_small_net_5000.tsv',
            (test_labels // 10 * 10).T, delimiter='\t', fmt='%f')

    val_features = []
    for i in range(len(val_imgs)):
        val_features.append(model2.predict(val_imgs[i:i+1]).flatten())
    val_features = np.array(val_features)
    print("val_features.shape:{}".format(val_features.shape))
    np.savetxt('val_embeddings_small_net_5000.tsv', 
            val_features, delimiter='\t')
    np.savetxt('val_labels_small_net_5000.tsv', 
            (val_labels // 10 * 10).T, delimiter='\t', fmt='%f')

    import bpython
    bpython.embed(locals())

def small_net(width, height, depth, train_imgs, train_labels, val_imgs, val_labels):
    num_classes = train_labels.shape[1]
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

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)
    #es = EarlyStopping(monitor='val_mean_squared_error', mode='auto', verbose=1, patience=100, restore_best_weights=True)

    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics=['mse','mae'])
    #model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    #checkpoint = ModelCheckpoint(str(fine_tune)+str(rand_init)+ str(training) + network + 'epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', 
    checkpoint = ModelCheckpoint('epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', 
            verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

    batch_size = 128
    history = model.fit(
        train_imgs,
        train_labels,
        batch_size=batch_size,
        epochs=4,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_imgs, val_labels),
        callbacks=[checkpoint]#, es]
    )
    # return the constructed network architecture
    return model

def random_forest_regresor(inp, model, train_imgs, train_labels,val_imgs, val_labels, test_imgs, test_labels, fine_tune, rand_init, training, network):
    model2 = Model(inputs=model.input, outputs=model.layers[-3].output)

    train_features = []
    for i in range(len(train_imgs)):
        train_features.append(model2.predict(train_imgs[i:i+1]).flatten())
    train_features = np.array(train_features)
    np.savetxt('embeddings.tsv', train_features, delimiter='\t')
    print(train_features.shape)
    regr = RandomForestRegressor()
    regr.fit(train_features, train_labels)

    val_features = []
    for i in range(len(val_imgs)):
        val_features.append(model2.predict(val_imgs[i:i+1]).flatten())
    val_features = np.array(val_features)
    print("random forest val r2 score: {}".format(regr.score(val_features, val_labels)))

    test_features = []
    for i in range(len(test_imgs)):
        test_features.append(model2.predict(test_imgs[i:i+1]).flatten())
    test_features = np.array(test_features)
    print("random forest test r2 score: {}".format(regr.score(test_features, test_labels)))

    with open('RFreg_modelSeed42'+ str(fine_tune)+str(rand_init)+ str(training) + network, 'wb') as f:
        pickle.dump(regr, f)

def create_model(train_imgs, train_labels, val_imgs, val_labels, fine_tune, rand_init, training, network):
    #base_model = Sequential()
    inp = keras.layers.Input((base_model_img_size , base_model_img_size , 3))

    if network == 'resnet':
        if rand_init != False :
            base_model = ResNet50(include_top = False, weights=None,
                    input_tensor = inp, input_shape = (base_model_img_size,base_model_img_size, 3))
        elif rand_init == False :
            base_model = ResNet50(include_top = False, weights='imagenet', 
                    input_tensor = inp, input_shape = (base_model_img_size,base_model_img_size, 3))
    elif network == 'vgg':
        if rand_init != False :
            base_model = VGG16(include_top = False, weights=None,
                    input_tensor = inp, input_shape = (base_model_img_size,base_model_img_size, 3))
        elif rand_init == False :
            base_model = VGG16(include_top = False, weights='imagenet',
                    input_tensor = inp, input_shape = (base_model_img_size,base_model_img_size, 3))

    if fine_tune == True:
        for layers in base_model.layers[:]:
            layers.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.1)(x)
    out = keras.layers.Dense(1, activation= 'linear')(x)
    model = Model(inp, out)
    if training == False:
        return model

    sgd = optimizers.SGD(lr=0.004, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100)

    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics=['mse','mae'])
    #model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    #checkpoint = ModelCheckpoint(str(fine_tune)+str(rand_init)+ str(training) + network + 'epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', 
    checkpoint = ModelCheckpoint(network + 'epoch_-{epoch:03d}-_loss_{loss:03f}-_val_loss_{val_loss:.5f}.h5', 
            verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

    batch_size = 128
    history = model.fit(
        train_imgs,
        train_labels,
        batch_size=batch_size,
        epochs=400,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_imgs, val_labels),
        callbacks=[checkpoint, es]
    )
    return model



## Read the train data
X, Y = get_data(train_data_paths)
#X, Y = get_data('data/100000')
print("X.shape {}, Y.shape {}".format(X.shape, Y.shape))

Y = keras.utils.to_categorical(Y, num_classes=Y.max() + 1)
#labels = keras.utils.to_categorical(list(Y), num_classes=train_labels.max() + 1)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(X, Y,test_size=0.3, random_state=42)
print(train_data_paths+'_test')
test_imgs, test_labels = get_data(train_data_paths+'_test')
test_labels = keras.utils.to_categorical(test_labels , num_classes=test_labels.max() + 1)
#test_imgs, test_labels = get_data('data/5000')


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
                    #model = create_model(train_imgs, train_labels, val_imgs, val_labels, fine_tune=train_type, rand_init=init, training=mode, network=net)
                    model = small_net(base_model_img_size, base_model_img_size, 3, train_imgs, train_labels, val_imgs, val_labels)
                    cnn_eval(model, test_imgs, test_labels, val_imgs, val_labels)
                    #random_forest_regresor(inp, model, train_imgs, train_labels,val_imgs, val_labels, test_imgs, test_labels, fine_tune=train_type, rand_init=init, training=mode, network=net)
