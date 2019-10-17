# 1.

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


data = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data


def inspect(images, labels=None):
    plt.figure(figsize=(18, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
    plt.show()
    if labels is not None:
        print(labels[:5])


print(train_images.shape, train_labels.shape, test_images.shape,
      test_labels.shape)
inspect(train_images, train_labels)
inspect(test_images, test_labels)
print(train_images[0].max(), train_images[0].min())

data = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
print(train_images[0].max(), train_images[0].min())
inspect(train_images, train_labels)
inspect(test_images, test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(test_images, test_labels))

predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)
print(test_labels[:10], predictions[:10])

wrong = test_labels != predictions
false_images = test_images[wrong]
false_labels = test_labels[wrong]
false_predictions = predictions[wrong]

for i in range(5):
    plt.imshow(false_images[i])
    plt.show()
    print(false_labels[i], false_predictions[i])


def play(image, label):
    print(label)
    predictions = model.predict(tf.reshape(image, (1, 28, 28)))
    predictions = tf.reshape(predictions, (10, ))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(10), predictions)
    plt.show()


for image, label in zip(test_images[:10], test_labels[:10]):
    play(image, label)

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(test_images, test_labels))
