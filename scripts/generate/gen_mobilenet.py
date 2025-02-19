import numpy as np
import tensorflow as tf
import keras
import tensorflow_datasets as tfds

total_classes = 10

def load_data():
    result = tfds.load('cifar10', batch_size = -1)
    (x_train, y_train) = result['train']['image'],result['train']['label']
    (x_test, y_test) = result['test']['image'],result['test']['label']
    
    #x_train = x_train.numpy().astype('float32') / 256
    #x_test = x_test.numpy().astype('float32') / 256
    #x_train = tf.keras.applications.EfficientNetV2S.preprocess_input(x_train)
    #x_test = tf.keras.applications.EfficientNetV2S.preprocess_input(x_test)
    
    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=total_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=total_classes)
    return ((x_train, y_train), (x_test, y_test))

base_model = tf.keras.applications.MobileNet(
    #weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(32,32,3),
    include_top=False,
    pooling="avg",
)  # Do not include the ImageNet classifier at the top.

print(base_model.layers[2].get_weights())

base_model.trainable = True

inputs = keras.Input(shape=(32, 32, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(total_classes,activation="softmax")(x)

model = keras.Model(inputs, outputs)

(x_train, y_train), (x_test, y_test) = load_data()

sgd = tf.keras.optimizers.SGD(
    learning_rate = 0.1,
    momentum = 0.9,
    nesterov=True,
    weight_decay=1e-5)

print(base_model)

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])   
model.fit(x_train, y_train, batch_size=256, epochs=100, validation_data=(x_test,y_test), shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
