#by Phil Ross

import tensorflow as tf
import os
import pathlib
import numpy as np
import time

# Specify the TensorFlow model, labels, and classifiers

class_names = ["Cont Walk", "Foraging", "Other", "Resting", "Rooting", "Running", "Standing", "Trotting", "Vigilance"]
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'boar_model_basic.tflite')
input_file = os.path.join(script_dir, 'input_array.npy')
output_file = os.path.join(script_dir, 'output_array.npy')
#input_tensor = tf.convert_to_tensor(np.load(input_file))
input_np = np.load(input_file)

output_np = np.load(output_file)
output_np = np.array(output_np).astype(int)
print("Output Numpy")
print(output_np)

#allocate tensors
interpreter = tf.lite.Interpreter(model_file)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:")
print(input_details)

input_shape = input_details[0]['shape']
print(input_shape)
numer =0
dem =13191

interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
input_data = np.array(input_np, dtype = np.float32)

invoke_time_total = 0
print('Beginning Inferencing...')
start = time.perf_counter()

for i in range(13191):
    #make sure to expand the dimensions- t
    input_data_new = np.expand_dims(input_data[i%13191], 0)
    interpreter.set_tensor(input_details[0]['index'], input_data_new)
    invoke_time_start = time.perf_counter()
    interpreter.invoke()
    invoke_time = time.perf_counter() - invoke_time_start
    invoke_time_total+=invoke_time
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print("Prediction: " + class_names[np.argmax(output_data)])
    #print("Actual: " +class_names[output_np[i]])

    if class_names[np.argmax(output_data)] == class_names[output_np[i%13191]]:
        numer += 1

inference_time = time.perf_counter()-start
print('------RESULTS------')
print('Time: %.1fms' % (inference_time *1000))
print('Invoke Time: %.1fms' % (invoke_time_total *1000))
print("Accuracy: " + str(numer*100/dem) + "%")
