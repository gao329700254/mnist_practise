import tensorflow as tf
import numpy as np
import os
from PIL import Image

def resize_img(file_path):
  img = Image.open(file_path)
  img_resize = img.resize((180, 180))
  img_resize.save(file_path)
