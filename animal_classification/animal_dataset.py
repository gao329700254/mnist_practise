import pathlib
import tensorflow as tf

data_root = '/Users/yuhaogao/workspace/tf_practise/animal_classification/data'
img_height = 180
img_width = 180
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

class AnimalDataSet:
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
            color_mode='grayscale',
            image_size=(img_height, img_width),
            batch_size=batch_size
        )
