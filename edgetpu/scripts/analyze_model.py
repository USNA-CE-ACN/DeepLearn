import tensorflow as tf
import sys

tf.lite.experimental.Analyzer.analyze(model_path=sys.argv[1])
