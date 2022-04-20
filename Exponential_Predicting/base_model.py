from operator import mod
from statistics import mode
from tabnanny import verbose
import numpy as np
import tensorflow
import keras
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from generate_data import data


# Regression testing
l_val = 7
dataloader = data(l_val)
drop_vals = np.arange(len(dataloader.columns)-1).tolist()
x_train, x_valid, y_train, y_valid = train_test_split(dataloader.drop(columns = drop_vals).to_numpy(), 
                                                      dataloader.drop(columns='c').to_numpy(), test_size=0.2, 
                                                      random_state=0)
def load_model(): #Function to load model upon calling
    model = Sequential()
    model.add(Dense(1, activation="relu")) #Attempts to avoid overfitting data
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(7))
    return model

model = load_model() #Initiate model
keras.backend.clear_session() #Clear model backend and set random state
np.random.seed(42)
tensorflow.random.set_seed(42)
model.compile(optimizer='Adam', 
              loss=['mean_squared_error'],
              metrics=[keras.metrics.Accuracy()])

history = model.fit(x_train, [y_train], epochs=60, batch_size=20,
                    verbose=0) #Fit model
print(history) #Show model training

predictions = model.predict(x_valid)
results = model.evaluate(x_valid, y_valid, verbose=0)
acc = tensorflow.metrics.binary_accuracy(y_valid, predictions)
def plot_loss(history):
  plt.plot(history.history['loss'], label='mse')
  plt.ylim([0, 0.00001])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  return plt.show()

if __name__ == '__main__':
    plot_loss(history)
