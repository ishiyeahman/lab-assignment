import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2

import tensorflow.keras as keras
from keras.utils import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
    

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import json
import time

time_stamp = time.time()

def get_all_data(path):
    files = glob.glob(path + '*.png')
    files = sorted(files)
    images = list( map( lambda x: cv2.imread(x), files ) )
    return images

def get_data_set(dir_path):
    healthy = get_all_data( dir_path + '0/' )
    unhealthy = get_all_data( dir_path + '1/' )

    x_data = np.array( healthy + unhealthy )
    y_data = np.array( [0] * len(healthy) + [1] * len(unhealthy) )

    x_data = normalize_data(x_data)
    y_data = to_categorical(y_data, 2)

    return x_data, y_data

def normalize_data(x_data):
    x_data = x_data.astype('float32')
    x_data /= 255
    return x_data

def model(X, y):
    model = Sequential() 
    model.add(Flatten())
    model.add(Dense(64, activation='relu', input_shape=(256,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    opt = keras.optimizers.SGD(learning_rate=0.001, decay=1e-6)
    
    print("---")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def plot_history(history):
    plt.plot(history.history['accuracy'],"o-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'out/acc-{time_stamp}.pdf')
    plt.cla()
 
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'out/loss-{time_stamp}.pdf')
    plt.cla()

def main():
    X_train, y_train = get_data_set('train/') 
    X_val, y_val  =  get_data_set('val/')


    # print("##########")
    # modelCheckpoint = ModelCheckpoint(filepath = 'direction_4.hdf5',
    #                                 monitor='val_loss', # 監視する値
    #                                 verbose=1, # 結果表示の有無
    #                                 save_best_only=True, 
    #                                 save_weights_only=True,
    #                                 mode='min', 
    #                                 )

    callback = EarlyStopping(monitor='val_loss', patience=10) #, verbose=1, mode='auto')
    mdl = model(X_train, y_train)
    history = mdl.fit(X_train, y_train, 
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        # callbacks=[callback]
        )
    

    mdl.save(f'model-{time_stamp}.h5')
    print(mdl.summary())
    plot_history(history)

    X_test, y_test  =  get_data_set('test/')

    results = mdl.evaluate(X_test, y_test)
    print('test loss, test acc:', results)
    predictions = mdl.predict(X_test)
    print(predictions)

    #AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], predictions[:,1])
    auc = metrics.auc(fpr, tpr)
    print('AUC: %.3f' % auc)
    
    #ROC
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig(f'out/roc-{time_stamp}.pdf')
    plt.cla()
    # plt.show()


    
if __name__ == '__main__':
    main()