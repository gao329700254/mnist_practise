import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import *

class MnistFashionNet:
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def evaluate(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\nTest accuracy:', self.test_acc)

    def predict_all(self):
        self.predictions = self.model.predict(self.test_images)

    def show_input_images(self):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap = plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.show()

    def show_input_images_and_predicts(self):
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
          plt.subplot(num_rows, 2*num_cols, 2*i+1)
          self.plot_image(i, self.predictions, self.test_labels, self.test_images)
          # net.plot_image(0, net.predictions, net.test_labels, net.test_images)

          plt.subplot(num_rows, 2*num_cols, 2*i+2)
          self.plot_value_array(i, self.predictions, self.test_labels)
        plt.show()

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        self.class_names[true_label]),
                                        color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def show_input_image_and_predict(self):
        prediction = self.model.predict(np.expand_dims(self.test_images[0], 0))
        self.plot_value_array(0, prediction, self.test_labels)
        _ = plt.xticks(range(10), self.class_names, rotation=45)
        plt.show()

net = MnistFashionNet()
net.fit()
net.evaluate()
net.predict_all()
# net.show_input_images_and_predicts()
# net.show_input_image_and_predict()

import pdb; pdb.set_trace()
