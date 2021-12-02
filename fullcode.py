import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import cv2
#from tensorflow.keras import layers,datasets,models
import ssl
import os

test=cv2.imread(r'C:\Users\HP\Downloads\diksha.jpg')
test_img=cv2.resize(test,(256,256))
test_img.resize(1,256,256,3)

labels=['diksha','rohit']

#first diksha folder is opened then rohit

images=[]
for filename in os.listdir(r"D:\dataset"):
    for file in os.listdir(os.path.join(r"D:\dataset",filename)):
        img=cv2.imread(os.path.join(os.path.join(r"D:\dataset",filename),file))
    #print(img.shape)
        if img is not None:
            temp=cv2.resize(img,(256,256))
            images.append(temp)
    


x_train=np.array(images)
#cv2.imshow('k',x_train[0])
y=np.array([[0],[1]])
y_train=np.repeat(y,9)
y_train.resize(18,1)


#x_train=x_train/255
#y_train=y_train/255

model=keras.Sequential([
                  keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
                  keras.layers.MaxPooling2D(2,2),

                  keras.layers.Conv2D(filters=300,kernel_size=(3,3),activation='relu'),
                  keras.layers.MaxPooling2D(2,2),

                  keras.layers.Flatten(),

                  keras.layers.Dense(200,activation='relu'),
                  keras.layers.Dense(100,activation='relu'),
                  keras.layers.Dense(2,activation='softmax')

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
print(model.predict(test_img))
print(labels[np.argmax(model.predict(test_img))])



