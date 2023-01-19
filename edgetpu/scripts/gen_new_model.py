import sys,os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

class Operation:
    def __init__(self):
        self.layer_type = ""
        # Tensor numbers for input, shape (Conv2D), and output
        self.number = 0
        self.input = 0
        self.filter = 0
        self.output = 0
        self.stride_w = 0
        self.stride_h = 0
        self.padding = ""
        self.activation = ""
        
ops = []
tensors = []

input_file = sys.argv[1]

os.system("python3 analyze_model.py " + input_file + " > model.analysis")
# Outputs (sys.argv[1] - .tflite) + .json in local directory
#os.system("flatc -t --strict-json --defaults-json schema.fbs -- " + sys.argv[1])

for line in open("model.analysis","r"):
    m = re.match("\s+Op\#(\d+)\s+([A-Z,a-z,0-9,\_,\-]+)\(T\#(\d+),\s(.*)\s\-\>\s\[T\#(\d+)\].*",line)
    if m:
        op = Operation()
        op.number = int(m.group(1))
        op.layer_type = m.group(2)
        op.input = int(m.group(3))
        
        remainder = m.group(4)

        op.output = int(m.group(5))
        
        if op.layer_type == "CONV_2D":
            m = re.match("T\#(\d+)\,.*",remainder)
            op.filter = int(m.group(1))
        elif op.layer_type == "CONCATENATION":
            # Switch input to list of inputs
            tmp = op.input
            op.input = [tmp]

            remainder = remainder[0:remainder.index(")")]
            inputs = remainder.split(",")
            for i in inputs:
                m = re.match("\s*T\#(\d+).*",i)
                op.input.append(int(m.group(1)))

        ops.append(op)

    m = re.match("\s+T\#(\d+).*shape\:\[([0-9,\,,\s]+)\].*",line)
    if m:
        shape = tuple(map(int, m.group(2).split(", ")))
        tensors.append(shape)

sys.exit(1)
    
input_size = 100
X_full = np.random.rand(100,1,224,224,3)
rng = np.random.default_rng()
Y_full = rng.integers(0,1000,100)

print(X_full.shape)

def representative_data_gen():
  for i in range(100):
    yield [X_full[i].astype(np.float32)]

keras.backend.clear_session()

### TODO: Create this model based on the input tflite model ###

input = (1,224,224,3)

a = tf.keras.layers.Input(shape=input[1:])

#x = keras.layers.InputLayer(input_shape=input)
b = tf.keras.layers.Conv2D(64,7,(2,2),padding="same",activation="relu",input_shape=input)(a)
c = tf.keras.layers.MaxPool2D((3,3),(2,2),padding="same",input_shape=b.shape)(b)
d = tf.keras.layers.Conv2D(64,1,(1,1),padding="same",activation="relu",input_shape=c.shape)(c)
e = tf.keras.layers.Conv2D(192,3,(1,1),padding="same",activation="relu",input_shape=d.shape)(d)

mp = keras.layers.MaxPool2D((3,3),(2,2),padding="same",input_shape=(1,56,56,192))(e)

c1 = keras.layers.Conv2D(64,1,(1,1),padding="same",activation="relu",input_shape=(1,28,28,192))(mp)

c2 = keras.layers.Conv2D(96,1,(1,1),padding="same",activation="relu",input_shape=(1,28,28,192))(mp)
c2 = keras.layers.Conv2D(128,3,(1,1),padding="same",activation="relu",input_shape=(1,28,28,96))(c2)

c3 = keras.layers.Conv2D(16,1,(1,1),padding="same",activation="relu",input_shape=(1,28,28,192))(mp)
c3 = keras.layers.Conv2D(32,3,(1,1),padding="same",activation="relu",input_shape=(1,28,28,16))(c3)

c4 = keras.layers.MaxPool2D((3,3),(1,1),padding="same",input_shape=(1,28,28,192))(mp)
c4 = keras.layers.Conv2D(32,1,(1,1),padding="same",activation="relu",input_shape=(1,28,28,192))(c4)

m = keras.layers.Concatenate()([c1,c2,c3,c4])

model = tf.keras.models.Model(inputs=a,outputs=m)

adam = keras.optimizers.Adam(epsilon = 1e-08)
model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#history = model.fit(X_full, Y_full, batch_size=128, epochs=1)
converter = tf.lite.TFLiteConverter.from_keras_model(model) #this works!!!!

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open("test.tflite", 'wb') as f:
   f.write(tflite_model)
