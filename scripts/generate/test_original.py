import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Reshape
import numpy as np
import re
import copy

import tensorflow_datasets as tfds

from utility.create_model import *
from utility.parse_tf_json import *
from utility.parse_tf_analysis import *

def load_data():
    result = tfds.load('imagenet_v2', batch_size=-1)
    print("Finished load")
    (x_test, y_test) = result['test']['image'],result['test']['label']
        
    new_x = []
    new_y = []
    
    for i in range(len(x_test)):
        image = tf.cast(x_test[i], tf.float32)
        image = tf.image.resize_with_crop_or_pad(image,224,224)
        #image = tf.keras.applications.mobilenet.preprocess_input(image)
        new_x.append(image)
        new_y.append(y_test[i])

    x_test = tf.convert_to_tensor(new_x,dtype=tf.uint8)
    y_test = tf.convert_to_tensor(new_y)
    y_test = tf.keras.utils.to_categorical(y_test,num_classes=1000)
    
    return (x_test,y_test)

base_model = tf.keras.applications.MobileNet(
)  # Do not include the ImageNet classifier at the top. 

base_model.trainable = False

print("Loading Data")

(test_x, test_y) = load_data()

print("Loaded data, compiling model")

base_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
#base_model.fit(train_x, train_y, batch_size=256, epochs=25, validation_data=(x_test,y_test), shuffle=True)

#print(base_model.summary())

#print(test_x[0])

scores = base_model.evaluate(test_x,test_y,verbose=1)
print('Test loss:', scores[0])                                                                                                                                                                            
print('Test accuracy:', scores[1])
