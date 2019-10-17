# 3.

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


a = tf.Variable(2.)
b = tf.Variable(3.)
with tf.GradientTape() as tape:
    c = a * b
gradient = tape.gradient(c, b)
b.assign_sub(gradient * .1)
print(gradient, b)

optimizer = tf.keras.optimizers.Adam(.2)
a = tf.Variable(2.)
b = tf.Variable(3.)
with tf.GradientTape() as tape:
    c = a * b
gradient = tape.gradient(c, b)
optimizer.apply_gradients([(gradient, b)])
print(gradient, b)

data = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

model = tf.keras.models.load_model('cnn.h5')
model.summary()

loss = tf.keras.losses.sparse_categorical_crossentropy
true = tf.reshape(1, (1, ))
pred = tf.reshape([.4, .3, .2], (1, 3))
print(true, pred, loss(true, pred))
true = tf.reshape(1, (1, ))
pred = tf.reshape([0., 1., 0.], (1, 3))
print(true, pred, loss(true, pred))

optimizer = tf.keras.optimizers.Adam()
loss_fun = tf.keras.losses.sparse_categorical_crossentropy
original_image = train_images[:1]
label = np.array([8])
original_prediction = model(original_image)


def train(steps):
    image = tf.Variable(np.array(original_image))
    for _ in range(steps):
        with tf.GradientTape() as tape:
            prediction = model(image)
            loss = loss_fun(label, prediction)
        gradient = tape.gradient(loss, image)
        optimizer.apply_gradients([(gradient, image)])
    return image


def play(steps):
    print(steps)
    image = train(steps)
    prediction = model(image)

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(original_image[0])
    plt.subplot(1, 4, 2)
    plt.imshow(image[0])
    plt.subplot(1, 4, 3)
    plt.plot(np.arange(10), original_prediction[0])
    plt.subplot(1, 4, 4)
    plt.plot(np.arange(10), prediction[0])
    plt.show()
    return image


for steps in range(310, 320):
    image = play(steps)
