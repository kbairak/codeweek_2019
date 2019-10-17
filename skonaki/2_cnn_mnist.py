# 2.

import itertools
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


data = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

layer = tf.keras.layers.Conv2D(filters=1, kernel_size=5)
x = train_images[0]
x = tf.reshape(x, (1, 28, 28, 1))
y = layer(x)
x = tf.reshape(x, (28, 28))
y = tf.reshape(y, (24, 24))
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.subplot(1, 2, 2)
plt.imshow(y)

layers = [tf.keras.layers.Conv2D(filters=1, kernel_size=5) for _ in range(5)]


def play(x):
    x = tf.reshape(x, (1, 28, 28, 1))
    y = [layer(x) for layer in layers]
    x = tf.reshape(x, (28, 28))
    y = [tf.reshape(_y, (24, 24)) for _y in y]
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(x)
    for i in range(5):
        plt.subplot(1, 6, i + 2)
        plt.imshow(y[i])
    plt.show()


for image in train_images[:5]:
    play(image)

layer = tf.keras.layers.MaxPool2D(2)
x = train_images[0]
x = tf.reshape(x, (1, 28, 28, 1))
y = layer(x)
x = tf.reshape(x, (28, 28))
y = tf.reshape(y, (14, 14))
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.subplot(1, 2, 2)
plt.imshow(y)

L = tf.keras.layers


def create_model():
    model = tf.keras.Sequential([
        L.Reshape((28, 28, 1), input_shape=(28, 28)),
        L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        L.MaxPool2D((2, 2)),
        L.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        L.MaxPool2D((2, 2)),
        L.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        L.Flatten(),
        L.Dense(64, activation='relu'),
        L.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))

activation_model = tf.keras.Model(
    inputs=model.input, outputs=[layer.output for layer in model.layers]
)
x = train_images[:1]
y = activation_model(x)


def play(x):
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x)

    x = tf.reshape(x, (1, 28, 28))
    y = activation_model(x)

    # a = y[2]
    # a = tf.reshape(a, (13, 13, 32))
    # a = tf.transpose(a, (2, 0, 1))
    # c = itertools.count(1)
    # plt.figure(figsize=(18, 8))
    # for image in a:
    #     plt.subplot(4, 8, next(c))
    #     plt.imshow(image)
    # plt.show()

    # a = y[4]
    # a = tf.reshape(a, (5, 5, 64))
    # a = tf.transpose(a, (2, 0, 1))
    # c = itertools.count(1)
    # plt.figure(figsize=(18, 18))
    # for image in a:
    #     plt.subplot(8, 8, next(c))
    #     plt.imshow(image)
    # plt.show()

    # a = y[5]
    # a = tf.reshape(a, (3, 3, 64))
    # a = tf.transpose(a, (2, 0, 1))
    # c = itertools.count(1)
    # plt.figure(figsize=(18, 18))
    # for image in a:
    #     plt.subplot(8, 8, next(c))
    #     plt.imshow(image)
    # plt.show()

    # a = y[6]
    # a = tf.reshape(a, (24, 24))
    # plt.imshow(a)
    # plt.show()

    plt.subplot(1, 3, 2)
    a = y[7]
    a = tf.reshape(a, (8, 8))
    plt.imshow(a)

    plt.subplot(1, 3, 3)
    a = y[8]
    a = tf.reshape(a, (1, 10))
    plt.imshow(a)
    plt.show()


for i in range(10):
    play(train_images[i])

layer = model.layers[-1]
w, b = layer.variables
w = tf.transpose(w, (1, 0))
w = tf.reshape(w, (10, 8, 8))
w.shape
plt.figure(figsize=(18, 8))
c = itertools.count(1)
for v in w:
    plt.subplot(2, 5, next(c))
    plt.imshow(v)

model.save('cnn.h5')
model = create_model()
print(model.evaluate(test_images, test_labels, verbose=0))
model.load_weights('cnn.h5')
print(model.evaluate(test_images, test_labels, verbose=0))
