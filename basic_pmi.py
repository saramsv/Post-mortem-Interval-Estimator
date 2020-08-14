from keras.applications import ResNet50
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
import csv
import pickle


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

def cnn_eval(model, test_imgs, test_labels):
    import sklearn.metrics, math

    preds = model.predict(test_imgs)
    y_pred = []
    for p in preds:
        y_pred.append(p[0])
    y_test = test_labels
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
    print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))


def random_forest_regresor(inp, model, train_imgs, train_labels,val_imgs, val_labels, test_imgs, test_labels, fine_tune, rand_init, training, network):
    model2 = Model(inputs=model.input,outputs=model.layers[-3].output)

    train_features = []
    for i in range(len(train_imgs)):
        train_features.append(model2.predict(train_imgs[i:i+1]).flatten())
    train_features = np.array(train_features)
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
#X, Y = get_data('img2PMI_shuf_train')
X, Y = get_data('legsPMI_train')
print("X.shape {}, Y.shape {}".format(X.shape, Y.shape))
train_imgs, val_imgs, train_labels, val_labels = train_test_split(X, Y,test_size=0.3, random_state=42)
#test_imgs, test_labels = get_data('img2PMI_shuf_test')
test_imgs, test_labels = get_data('legsPMI_test')

## Run the model
networks = ['resnet']#, 'vgg']
rand_init = [False]#[True, False]
training = [True]#, False]
fine_tune = [False]#[True, False]
for net in networks:
    for init in rand_init:
        for mode in training:
            for train_type in fine_tune:
                if init == True and mode == True:
                    continue
                else:
                    print("network: {},rand init: {}, training: {}, fine_tune:{}".
                            format(net, init, mode, train_type))
                    model = create_model(train_imgs, train_labels, val_imgs, val_labels, fine_tune=train_type, rand_init=init, training=mode, network=net)
                    cnn_eval(model, test_imgs, test_labels)
                    random_forest_regresor(inp, model, train_imgs, train_labels,val_imgs, val_labels, test_imgs, test_labels, fine_tune=train_type, rand_init=init, training=mode, network=net)












