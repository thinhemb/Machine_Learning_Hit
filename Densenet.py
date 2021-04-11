import keras
import os
import cv2
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, AveragePooling2D, Activation, ZeroPadding2D,Flatten,Input,Add,Concatenate
from keras.utils import plot_model
from utils import *
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import BatchNormalization
(X_train, y_train), (X_test, y_test)= mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
X_train =X_train/255
X_test =X_test/255
def dense_layer(input_tensor,filter):
    """
    BN-RELU-Conv(1,1) _ BN_RELU_Conv(3,3)
    """
    out=BatchNormalization(axis=3)(input_tensor)
    out=Activation('relu')(out)
    out=Conv2D(filter*4,(1,1),strides=1,use_bias=False)(out)

    out=BatchNormalization(axis=3)(input_tensor)
    out=Activation('relu')(out)
    out-Conv2D(filter,(3,3),strides=1,padding='same')(out)
    output=Concatenate(axis=3)([out,input_tensor])
    return output
def dense_block(input_tensor,nb_layers):
    """
     a block of dense layers
    """
    for i in range(nb_layers):
        input_tensor = dense_layer(input_tensor,32)
    return input_tensor


def transition_block(input_tensor):
    filter=K.int_shape(input_tensor)[3]
    x=BatchNormalization(axis=3)(input_tensor)
    x=Activation('relu')(x)
    x=Conv2D(filter,(1,1))(x)
    x=AveragePooling2D(pool_size=(2,2),stride=2,padding='same')(x)
    return x

    '''
    Hàm khởi tạo model Densenet.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model Densenet với softmax
    '''
def Densenet(input_shape=(224,224,3),n_classes=10):
    In=Input(shape=input_shape)
    x=Conv2D(64,(7,7),strides=2,padding='same')(In)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)

    x=dense_block(x,6)
    x=transition_block(x)

    x=dense_block(x,12)
    x=transition_block(x)

    x=dense_block(x,24)
    x=transition_block(x)

    x=dense_block(x,16)

    x=keras.layers.GlobalAveragePooling2D()(x)
    x=Dense(n_classes,activation='softmax')(x)
    model=Model(inputs=In,outputs=x)
    return model


model=Densenet((28,28,1),n_classes=10)
model.compile(optimizer='Adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy']
            )
model.fit(X_train, y_train, batch_size=8, epochs= 10, validation_data=(X_test, y_test))
model.save("Densenet.h5")