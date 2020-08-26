import pathlib
import os
import pprint
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from pprint import pprint as pp
from tensorflow.keras import layers
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
img_height = 180
img_width = 180
data_root = '/Users/yuhaogao/workspace/tf_practise/dog_classification/data'
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

    def fit_and_save_weight(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        self.fit(5, cp_callback)

    def load_weight(self):
        self.model.load_weights(checkpoint_path)

class AnimalData:
    def __init__(self):
        self.train_ds = self.image_dataset_from_directory("training")
        self.val_ds = self.image_dataset_from_directory("validation")

        self.class_names = self.train_ds.class_names

        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def image_dataset_from_directory(self, subset):
        return tf.keras.preprocessing.image_dataset_from_directory(
            pathlib.Path(data_root),
            validation_split=0.2,
            subset=subset,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [img_width, img_height])
  return image

animal_data = AnimalData()
animal_net = AnimalNet(animal_data)
animal_net.load_weight()

all_paths = [str(path) for path in list(pathlib.Path(data_root).glob('*/*'))]
random.shuffle(all_paths)
all_paths = all_paths[:int(len(all_paths) * 0.2)]

label_to_index = dict((name, index) for index,name in enumerate(animal_data.class_names))
all_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_paths]

path_ds = tf.data.Dataset.from_tensor_slices(all_paths)
image_ds = path_ds.map(load_and_preprocess_image).batch(len(path_ds)).prefetch(buffer_size=AUTOTUNE)

preds = [ tf.math.argmax(pred).numpy() for pred in animal_net.model(next(iter(image_ds))) ]

diff = [ [all_paths[i], all_labels[i], pred] for i, pred in enumerate(preds) if all_labels[i] != pred ]
import pdb; pdb.set_trace()

# predictions = [ animal_net.predict(np.array([image])) for image in image_ds ]

# val_ds = animal_data.image_dataset_from_directory('validation')

# all_imgs = []
# all_labels = []
# for batch, labels in iter(val_ds):
#     all_imgs.extend(batch.numpy())
#     all_labels.extend(labels.numpy())

# predictions = animal_net.model.predict(np.array(all_imgs))
# predictions_labels = np.apply_along_axis(lambda preds: np.argmax(preds), 1, predictions).tolist()

# diff = [ True if predictions_labels[i] == label else False for i, label in enumerate(all_labels) ]


# val_dsのlabel配列を取得
# val_dsの推論配列を取得
# (path, label, predict)のタップル配列を作る

# TODO: validationで誤答のものを見つけて、対策(e.g., 削除か、データを増やす、drop層追加)を打つ
    # 怪しいデータを削除
    # drop層追加


# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
