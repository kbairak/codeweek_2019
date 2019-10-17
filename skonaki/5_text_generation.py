# 5.

import tensorflow as tf


path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'  # noqa
)
with open(path_to_file) as f:
    text = f.read()
    print(text[:250])

idx2char = sorted(set(text))
char2idx = {c: i for i, c in enumerate(idx2char)}
list(char2idx.items())[:5]

text_as_int = [char2idx[c] for c in text]
text_as_int[:10]

SEQUENCE_LENGTH, BATCH_SIZE, BUFFER_SIZE = 100, 64, 10_000
dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
dataset = dataset.batch(SEQUENCE_LENGTH + 1, drop_remainder=True)
dataset = dataset.map(lambda s: (s[:-1], s[1:]))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
next(iter(dataset.take(1)))

L = tf.keras.layers
EMBEDDING_SIZE = 8
RNN_SIZE = 512


def get_model(batch_size=BATCH_SIZE):
    return tf.keras.Sequential([
        L.Embedding(len(idx2char), EMBEDDING_SIZE,
                    batch_input_shape=(batch_size, None, )),
        L.GRU(RNN_SIZE, return_sequences=True, stateful=True),
        L.Dense(len(idx2char)),
    ])


model = get_model()
model.summary()

batch = next(iter(dataset.take(1)))
predictions = model(batch)
predictions = tf.argmax(predictions, -1)
predictions = predictions[0]
sentence = "".join((idx2char[c.numpy()] for c in batch[0][0]))
prediction = "".join((idx2char[c.numpy()] for c in predictions))
sentence, prediction

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
true = tf.reshape([1], (1, ))
pred = tf.reshape([.8, .1, -3.4], (1, 3))
loss_func(true, pred)

model.compile(optimizer="adam", loss=loss_func,
              metrics=['sparse_categorical_accuracy'])
model.fit(dataset, epochs=100)

# model = tf.keras.models.load_model('text_generation_backup.h5')
# model.save('text_generation_backup.h5')
# model.load_weights('text_generation.h5')

model.save('text_generation.h5')
model = get_model(1)
model.load_weights('text_generation.h5')
model.summary()
batch = next(iter(dataset.take(1)))
sentence = batch[0][:1]
predictions = model(sentence)
predictions = tf.argmax(predictions, -1)
sentence = "".join((idx2char[i] for i in sentence[0]))
predictions = "".join((idx2char[i] for i in predictions[0]))
sentence, predictions


def generate(start_string="ROMEO: ", length=100, temperature=1.):
    start = [char2idx[c] for c in start_string]
    start = tf.reshape(start, (1, -1))
    model.reset_states()
    for _ in range(length):
        prediction = model(start)
        prediction = prediction[:, -1, :]
        # prediction = tf.argmax(predictions, -1)
        prediction = tf.random.categorical(prediction / temperature, 1)
        start = tf.reshape(prediction, (1, 1))
        start_string += idx2char[tf.reshape(prediction, ()).numpy()]
    return start_string


print(generate(temperature=.1))
