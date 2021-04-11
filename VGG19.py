import keras
import os
import cv2
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, AveragePooling2D, Activation, ZeroPadding2D,Flatten,Input,Add,concatenate
from keras.models import Sequential,Model
from keras.utils import plot_model
from utils import *
from keras.datasets import mnist
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test)= mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
X_train =X_train/255
X_test =X_test/255
'''
    Hàm khởi tạo model VGG16.
    đầu vào: kích thước của ảnh.(width,height,channel)
    đầu ra: model VGG19 với softmax 
'''
def VGG19(input_shape=(224,224,3),n_classes=10):
    In=Input(shape=input_shape)
    #block 1 start here
    x=Conv2D(64,(3,3),padding='same')(In)
    x=Activation('rule')(x)
    x=Conv2D(64,(3,3),padding='same')(x)
    x=Activation('rule')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)

    #block 2 start here
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('rule')(x)
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    #block 3 start here
    x=Conv2D(256,(3,3),padding='same')(x)
    x=Activation('rule')(x)
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 4 start here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 5 start here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)

    # Flatten and fully connected layer start here
    x=Flatten()(x)
    x=Dense(units=25088,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=n_classes,activation='softmax')(x)
    model=Model(In,x)
    return model
model=VGG19((28,28,1),n_classes=10 )
model.compile(optimizer='Adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy']
            )
model.fit(X_train, y_train, batch_size=8, epochs= 10, validation_data=(X_test, y_test))
model.save("VGG19.h5")