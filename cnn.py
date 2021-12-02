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
