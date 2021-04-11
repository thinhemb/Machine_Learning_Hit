import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model
(X_train, y_train), (X_test, y_test)= mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
X_train =X_train/255
X_test =X_test/255

model = Sequential()
model.add(Input((28,28,1)))
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(256, (3,3), 1, activation='relu'))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

X_train = np.reshape(X_train, (60000, 28,28,1))
X_test = np.expand_dims(X_test, axis = -1)
print(X_test.shape, " ", X_train.shape)

model.compile(optimizer='Adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy']
            )

model.fit(X_train, y_train, batch_size=8, epochs= 10, validation_data=(X_test, y_test))
model.save("fuck.h5")