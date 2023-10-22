import os
import tensorflow as tf
import cv2
import numpy as np
import imghdr
import matplotlib.pyplot as plt
# Avoid OOM error
# cpus = tf.config.experimental.list_physical_devices('CPU')
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define data directory and image extensions
data_dir = 'data'
image_exts = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

# Iterate through image classes and clean the data
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not in the list: {}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Remove error: {}".format(image_path))
            os.remove(image_path)

# Load and preprocess the data
data = tf.keras.preprocessing.image_dataset_from_directory('data')
data = data.map(lambda x, y: (x / 255.0, y))

# Split the data into train, validation, and test sets
train_size = int(len(data) * 0.65)
val_size = int(len(data) * 0.15) + 1
test_size = int(len(data) * 0.15) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Log training data to a directory
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model
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


# Save the trained model as an .h5 file
model.save('imgclassifer.h5')
