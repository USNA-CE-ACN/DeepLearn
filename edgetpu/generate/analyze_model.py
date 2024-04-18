from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import sys

tf.lite.experimental.Analyzer.analyze(model_path=sys.argv[1])
