import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np


class DrugAI:
    # creates neural network model
    def __init__(self, class_count, img_size, checkpoint_name):
        self.checkpoint_name = checkpoint_name
        self.checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_name,
            save_weights_only=True,
            verbose=1)
        self.img_size = img_size
        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)),
            layers.Conv2D(4, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(class_count)
        ])

    # compiles model
    def compile(self):
        self.model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.model.summary()

    # trains model
    def train(self, training_ds, validation_ds):
        self.history = self.model.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=10,
            callbacks=[self.checkpoint_callback]
        )

    # predicts what is on image, data_path is path to the image
    def predict(self, data_path):
        self.model.load_weights(self.checkpoint_name)

        class_names = ['antidol 15', 'apap noc', 'clatra', 'dexoftyal', 'etopiryna', 'gripex control', 'momester', 'octanisept', 'paracetamol synoptics', 'solpadeine']
        img = keras.preprocessing.image.load_img(
            data_path, target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        prediction = self.model.predict(img_array)[0]
        score = tf.nn.softmax(prediction)
        print("Processing image {}".format(data_path))
        print("This image is {} ({}% confidence)".format(class_names[np.argmax(score)], 100 * np.max(score)))
