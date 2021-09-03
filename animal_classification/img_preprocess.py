import tensorflow as tf
import numpy as np
import os
from PIL import Image

img_height = 180
img_width = 180

def resize_img(file_path):
    img = Image.open(file_path)
    img_resize = img.resize((180, 180))
    img_resize.save(file_path)

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [img_width, img_height])
    return image
