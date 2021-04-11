import keras
import os
import cv2
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, AveragePooling2D, Activation, ZeroPadding2D,Flatten,Input,Add,Concatenate,BatchNormalization,DepthwiseConv2D
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

def identity_block(input_tensor, output_channel,stage=1):
    re_c=64*(2**(stage-1))
    x=Conv2D(re_c,(1,1),strides=1,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    x=Conv2D(re_c,(3,3),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)

    x=Conv2D(output_channel,(1,1),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    
    x=Add()([x,input_tensor])
    x=Activation('relu')(x)
    return x

def conv_block(input_tensor, output_channel, stage=1):
    
    shortcut=input_tensor
    shortcut=Conv2D(output_channel,(1,1),strides=1,padding='same')(shortcut)
    shortcut=BatchNormalization(axis=3)(shortcut)
    


    re_c=64*(2**(stage-1))
    x=Conv2D(re_c,(1,1),strides=1,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    x=Conv2D(re_c,(3,3),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)

    x=Conv2D(output_channel,(1,1),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    
    x=Add()([x,shortcut])
    x=Activation('relu')(x)
    return x


def ResNet50(input_shape=(224,224,3), n_classes=10):
    '''
        Hàm khởi tạo model ResNet50.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model ResNet50 với softmax
    '''
    In=Input(shape=input_shape)
    x=Conv2D(64,(7,7),strides=2,padding='same')(In)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #stage 1 starts here
    x=conv_block(input_tensor=x,output_channel=256,stage=1)
    x=identity_block(input_tensor=x,output_channel=256,stage=1)
    x=identity_block(input_tensor=x,output_channel=256,stage=1)

    #stage 2 starts here
    x=conv_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)

    #stage 3 starts here
    x=conv_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)

    #stage 4 starts here
    x=conv_block(input_tensor=x,output_channel=2048,stage=2)
    x=identity_block(input_tensor=x,output_channel=2048,stage=2)

    x=AveragePooling2D(pool_size=(2,2))(x)
     
    #Fully connected
    x=Flatten()(x)
    x=Dense(units=n_classes,activation='softmax')(x)
    model=Model(In,x)
    return model

model=ResNet50((28,28,1),10)
model.compile(optimizer='Adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy']
            )
model.fit(X_train, y_train, batch_size=8, epochs= 10, validation_data=(X_test, y_test))
model.save("Resnet50.h5")