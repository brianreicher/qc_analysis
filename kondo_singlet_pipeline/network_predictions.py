import numpy as np
import tensorflow
import base_model

predictions = base_model.model.predict(base_model.x_valid)
results = base_model.model.evaluate(base_model.x_valid, base_model.y_valid, verbose=0)
acc = tensorflow.metrics.binary_accuracy(base_model.y_valid, predictions)