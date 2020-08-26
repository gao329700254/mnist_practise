
import pathlib
import random
import tensorflow as tf
import numpy as np
from pprint import pprint as pp

from animal_net import AnimalNet
from animal_dataset import AnimalDataSet

data_root = '/Users/yuhaogao/workspace/tf_practise/animal_classification/data'
img_height = 180
img_width = 180
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [img_width, img_height])
    return image

def diff():
    all_paths = [str(path) for path in list(pathlib.Path(data_root).glob('*/*'))]
    random.shuffle(all_paths)
    all_paths = all_paths[:int(len(all_paths) * 0.2)]

    label_to_index = dict((name, index) for index,name in enumerate(animal_dataset.class_names))
    index_to_label = dict((index, name) for index,name in enumerate(animal_dataset.class_names))
    all_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_paths)
    image_ds = path_ds.map(load_and_preprocess_image).batch(len(path_ds)).prefetch(buffer_size=AUTOTUNE)

    preds = [ tf.math.argmax(pred).numpy() for pred in animal_net.model(next(iter(image_ds))) ]

    diff = [ { 'img_path': all_paths[i], 'label': index_to_label[all_labels[i]], 'pred': index_to_label [pred] } for i, pred in enumerate(preds) if all_labels[i] != pred ]

    return diff

animal_dataset = AnimalDataSet()
animal_net = AnimalNet(animal_dataset)
animal_net.load_weight()

img_diff = diff()

import pdb; pdb.set_trace()
