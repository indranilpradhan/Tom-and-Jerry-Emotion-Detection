import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,BatchNormalization
from keras.optimizers import RMSprop, SGD
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.regularizers import l2
import csv

if(__name__=='__main__'):

    df = pd.read_csv('/content/dataset/db5fcca9-f52b-42a9-87be-26f77b6f9d97_train.csv')
    tdf = pd.read_csv('/content/final_test/c3b70bec-470d-4981-82ea-6e0693f2c8b0_test.csv')

    X_train = df.iloc[:,:-1]
    y_train = df.iloc[:,-1:]
    X_test = tdf

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i in range(X_train.shape[0]):
        ti_path = str(X_train.iloc[i][0])+'.jpg'
        tpath = os.path.join('/content/dataset/ntrain/',ti_path)
        tframe = cv2.imread(tpath)
        timg = cv2.cvtColor(tframe, cv2.COLOR_BGR2GRAY)
        #print('train ',timg.shape)
        train_label = y_train.iloc[i][0]
        train_images.append(np.asarray(timg).flatten())
        train_labels.append(train_label)

    for i in range(X_test.shape[0]):
        tei_path = str(X_test.iloc[i][0])+'.jpg'
        tepath = os.path.join('/content/final_test/test/',tei_path)
        teimg = cv2.imread(tepath)
        teimg = cv2.cvtColor(teimg, cv2.COLOR_BGR2GRAY)
        #test_label = y_test.iloc[i][0]
        dims = (640,360)
        teimg = cv2.resize(teimg,dims,interpolation = cv2.INTER_AREA)
        #print('test ',teimg.shape)
        test_images.append(np.asarray(teimg).flatten())
        #test_labels.append(test_label)

    fx_train = np.array(train_images).astype('float32')
    fx_test = np.array(test_images).astype('float32')
    fy_train = to_categorical(np.array(train_labels),5)
    fx_train = fx_train.reshape(fx_train.shape[0],360,640,1)
    fx_test = fx_test.reshape(fx_test.shape[0],360,640,1)

    model2 = Sequential()
    model2.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu',input_shape=(360,640,1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=256,kernel_size=(3, 3),activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=512,kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=512,kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(units=256, activation= 'relu'))
    model2.add(Dense(units=128, activation= 'relu'))
    model2.add(Dense(units=64, activation = 'relu'))
    model2.add(Dense(units=5, activation='softmax'))
    model2.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam', metrics=['accuracy'])
    model2.fit(x=fx_train,y=fy_train,batch_size=16, epochs=75)
    fpred3 = model2.predict_classes(fx_test)

    filename="submission.csv"
    with open(filename, "wb") as f:
        f.write(b'emotion\n')
        np.savetxt(f, fpred3.astype(int), fmt='%i', delimiter=",")