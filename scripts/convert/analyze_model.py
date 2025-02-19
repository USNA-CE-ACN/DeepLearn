import tensorflow as tf
import sys

#interpreter = tf.lite.Interpreter(model_path=sys.argv[1])

tf.lite.experimental.Analyzer.analyze(model_path=sys.argv[1])
