
import pathlib
import random
import tensorflow as tf
import numpy as np
from pprint import pprint as pp
from statistics import mean

from animal_net import AnimalNet
from animal_dataset import AnimalDataSet
from img_preprocess import *

data_root = '/Users/yuhaogao/workspace/tf_practise/animal_classification/data'
AUTOTUNE = tf.data.experimental.AUTOTUNE

def test_paths_for_diff():
    all_paths = [str(path) for path in list(pathlib.Path(data_root).glob('*/*'))]
    random.shuffle(all_paths)
    return all_paths[:int(len(all_paths) * 0.2)]

def diff_label_pred(paths):
    label_to_index = dict((name, index) for index,name in enumerate(animal_dataset.class_names))
    index_to_label = dict((index, name) for index,name in enumerate(animal_dataset.class_names))
    all_labels = [label_to_index[animal_dataset.class_name(path)] for path in paths]

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(load_and_preprocess_image).batch(len(path_ds)).prefetch(buffer_size=AUTOTUNE)

    probs = animal_net.probabilities(image_ds)
    preds = list(map(lambda class_probs: tf.math.argmax(class_probs).numpy(), probs))

    diff = [
        {
            'img_path': paths[i],
            'label': index_to_label[all_labels[i]],
            'pred': index_to_label [pred]
        }
        for i, pred in enumerate(preds) if all_labels[i] != pred
    ]

    return diff

def class_accuracy_mean():
    class_accuracies = []

    for _ in range(10):
        test_paths = test_paths_for_diff()

        test_labels = [ animal_dataset.class_name(path) for path in test_paths ]
        test_label_count = { class_name: test_labels.count(class_name) for class_name in animal_dataset.class_names }

        img_diff = diff_label_pred(test_paths)
        diff_label_count = {
            class_name: len([ diff for diff in img_diff if diff['label'] == class_name ])
            for class_name in animal_dataset.class_names
        }

        class_accuracies.append({
            class_name: 1 - diff_label_count[class_name] / test_count
            for class_name, test_count in test_label_count.items()
        })

    class_accuracy_mean = {
        class_name: mean([ cas[class_name] for cas in class_accuracies ])
        for class_name in animal_dataset.class_names
    }

    return class_accuracy_mean

animal_dataset = AnimalDataSet()
animal_net = AnimalNet(animal_dataset)
animal_net.load_weight()


import pdb; pdb.set_trace()

# 現在正解率 {'cat': 0.8838156632574367, 'dog': 0.9326540097346643, 'horse': 0.9632037177563963}
# TODO: テストデータセットもう一つ導入して、正解率を見よう
# TODO: data augmentation で正解率90%以上になるか試そう
