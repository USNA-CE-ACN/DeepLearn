from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model_file',
      default='dnn_baseline_nonorm_quant_edgetpu.tflite',
      help='.tflite model to be executed')
  args = parser.parse_args()

  interpreter = tflite.Interpreter(model_path=args.model_file,
                                   experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  input_np = np.load('input_array.npy')
  output_np = np.load('output_array.npy')

  input_data = np.array(input_np, dtype = np.float32)
  scale, zero_point = input_details[0]['quantization']
  input_data = input_data / scale
  input_data = input_data + zero_point

  idx = 0
  correct = 0
  total = 0
  invoke_time_total = 0
  
  for single_input in input_data:
      input_uint8 = np.uint8(single_input)
      input_tensor = np.expand_dims(input_uint8, axis=0)
      interpreter.set_tensor(input_details[0]['index'], input_tensor)
      
      invoke_time_start = time.perf_counter()
      interpreter.invoke()
      invoke_time = time.perf_counter() - invoke_time_start
      
      output_data = interpreter.get_tensor(output_details[0]['index'])
      results = np.squeeze(output_data)
      
      top_k = results.argsort()[-5:][::-1]

      if top_k[0] == output_np[idx]:
          correct = correct + 1

      idx = idx + 1
      invoke_time_total = invoke_time_total + invoke_time

  print("Accuracy: " + str(100*float(correct)/float(idx)) + "%")
  print("Time: " + str(invoke_time_total * 1000) + " ms")
