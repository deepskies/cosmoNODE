import demo as d
import macros as m

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

loader = d.Demo()

data = loader.merged

input_size = loader.input_size
output_size = loader.output_size

# "It's not required to set an input shape for the tf.keras.Model 
# class since the parameters are set the first time input is passed to the layer."

class MyModel(Model):
  def __init__(self, output_size):
    super(MyModel, self).__init__()
    #self.conv1 = Conv2D(input_size, 3, activation='relu')
    #self.flatten = Flatten()
    # self.l1 = Dense(64, activation='relu')
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(output_size, activation='softmax')

  def call(self, x):
    # x = self.conv1(x)
    # x = self.l1(x)
    # x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

target = data.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))

for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

train_dataset = dataset.shuffle(len(data)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model


model = get_compiled_model()

model.fit(train_dataset, epochs=2)

# functional implementation


# model = MyModel(output_size)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



# @tf.function
# def train_step(data, labels):
#   with tf.GradientTape() as tape:
#     predictions = model(data)
#     loss = loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#   train_loss(loss)
#   train_accuracy(labels, predictions)

# @tf.function
# def test_step(data, labels):
#   predictions = model(data)
#   t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)

# EPOCHS = 1

# for epoch in range(EPOCHS):
#   for data, labels in train_ds:
#     train_step(data, labels)

#   for test_data, test_labels in test_ds:
#     test_step(test_data, test_labels)

#   template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#   print (template.format(epoch+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          test_loss.result(),
#                          test_accuracy.result()*100))