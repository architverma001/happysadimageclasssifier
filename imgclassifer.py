import os
import tensorflow as tf
import cv2 as cv
import numpy as np
import imghdr
import matplotlib.pyplot as plt

# Avoid OOM error
# cpus = tf.config.experimental.list_physical_devices('CPU')
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = 'data'
image_exts = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
img = cv.imread(os.path.join(data_dir, 'happy', '35438_hd.jpg'))
# print(os.listdir(os.path.join(data_dir, 'happy')))
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not in the list: {}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Remove error: {}".format(image_path))
            os.remove(image_path)

data = tf.keras.preprocessing.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = next(data_iterator)
# print(batch[1])
data = data.map(lambda x,y:(x/255.0, y))
scaled_iterator = data.as_numpy_iterator()
batch = next(scaled_iterator)
print(len(data))
train_size = int(len(data)*0.7)
val_size = int(len(data)*0.15)+1
test_size = int(len(data)*0.15)+1
print(train_size, val_size, test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(val_size)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(16, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir) 
hist = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'],label='loss', c='red')
plt.plot(hist.history['val_loss'],label='val_loss', c='blue')
plt.legend()
plt.show()


plt.plot(hist.history['accuracy'],label='accuracy', c='red')
plt.plot(hist.history['val_accuracy'],label='val_accuracy', c='blue')
plt.legend()
plt.show()