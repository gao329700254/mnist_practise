import tensorflow as tf
from tensorflow.keras import layers

checkpoint_path = "training_1/cp.ckpt"

class AnimalNet:
    def __init__(self, data):
        self.train_ds = data.train_ds
        self.val_ds = data.val_ds
        self.class_names = data.class_names

        self.model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names))
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def fit(self, epochs=5, callbacks=[]):
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[callbacks]
        )

    def predict(self, image):
        self.model.predict(image)

    def evaluate(self, dataset):
        self.model.evaluate(dataset, verbose=2)

    def fit_and_save_weight(self, epochs=5):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        self.fit(epochs, cp_callback)

    def load_weight(self):
        self.model.load_weights(checkpoint_path)
