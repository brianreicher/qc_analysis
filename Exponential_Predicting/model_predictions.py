import tensorflow
from matplotlib import pyplot as plt
from base_model import x_train, x_valid, y_train, y_valid, history, model


predictions = model.predict(x_valid)
results = model.evaluate(x_valid, y_valid, verbose=0)
acc = tensorflow.metrics.binary_accuracy(y_valid, predictions)
def plot_loss(history):
  plt.plot(history.history['loss'], label='mse')
  plt.plot(history.history['loss'], label='mse')
  plt.plot(history.history['loss'], label='mse')
  plt.ylim([0, 0.00001])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  return plt.show()

if __name__ == '__main__':
    plot_loss(history)