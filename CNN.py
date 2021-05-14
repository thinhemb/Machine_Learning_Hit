import keras
import numpy as np
import tensorflow 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt
#load dataset
(X_train, y_train), (X_test, y_test)= mnist.load_data()
# mã hoá số  nguyên cho từng dối tượng
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
print('Dữ liệu y ban đầu ', y_train[1])
print('Dữ liệu y sau one-hot encoding ',y_train[1])
X_train =X_train/255
X_test =X_test/255
#Định nghĩa Model 
model = Sequential()
#thêm input chỉ rõ kích thước input
model.add(Input((28,28,1)))
#thêm convolution layer  với 64 kerel , kích thước của kerel là 3*3
#dùng hàm relu  làm hàm actication : mô phỏng tỷ lệ truyền xung qua axon của một neuron thần kinh
#hàm kích hoạt đóng vai trò là thành phần phi tuyến tại output của các nơ-ron
model.add(Conv2D(64, (3,3), 1, activation='relu'))

model.add(Conv2D(128, (3,3), 1, activation='relu'))
#Thêm Max pooling layer : r, để giảm kích thước dữ liệu nhưng vẫn giữ được các thuộc tính quan trọng 
# Kích thước dữ liệu giảm giúp giảm việc tính toán trong model
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(256, (3,3), 1, activation='relu'))
model.add(MaxPooling2D((2,2),strides=2))
#Flatten layer chuyển từ tensor sang vector
model.add(Flatten())
# Thêm Fully Connected layer với 1024 nodes và dùng hàm relu
model.add(Dense(1024, activation='relu'))
# Thêm Fully Connected layer với 10 nodes và dùng hàm softmax
model.add(Dense(10, activation='softmax'))

model.summary()

X_train = np.reshape(X_train, (60000, 28,28,1))
#Mở rộng hình dạng của một mảng. [1,1] -> [[1],[1]]
X_test = np.expand_dims(X_test, axis = -1)
print(X_test.shape, " ", X_train.shape)

# 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức
# đùng để tối ưu hàm loss function.

model.compile(optimizer='Adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy']
            )
#Thực hiện train model với data
model=model.fit(X_train, y_train, batch_size=8, epochs= 10, validation_data=(X_test, y_test))

#Vẽ đồ thị loss, accuracy của training set và validation set
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), model.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), model.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch),model.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), model.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
#Đánh giá model với dữ liệu test set
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

#. Dự đoán ảnh
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
y_predict = model.predict(X_test[0].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))


model.save("fuck.h5")

