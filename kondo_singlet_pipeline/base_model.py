import random
import numpy as np
import tensorflow
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from generate_data import data
from sklearn.model_selection import train_test_split

# Regression testing
c_range = random.randint(50, 60)
dataloader = data(c_range)
drop_vals = np.arange(len(dataloader.columns)-1).tolist()
x_train, x_valid, y_train, y_valid = train_test_split(dataloader.drop(columns=drop_vals).to_numpy(),
                                                      dataloader.drop(columns='c').to_numpy(), test_size=0.2, 
                                                      random_state=0)


def load_model():  # Function to load model upon calling
    seq = Sequential()
    seq.add(Dense(75, activation="relu"))  # Attempts to avoid over-fitting data
    seq.add(Dense(250, activation="relu"))
    seq.add(Dropout(0.2))
    seq.add(Dense(100, activation="relu"))
    seq.add(Dense(90, activation="relu"))
    seq.add(Dense(c_range))
    return seq


model = load_model()  # Initiate model
np.random.seed(42)
tensorflow.random.set_seed(42)

model.compile(optimizer='Adam',  # Compile model
              loss=['mse'],
              metrics=['accuracy'])  # Figure out hinge

history = model.fit(x_train, y_train, epochs=60, batch_size=20,
                    validation_data=(x_valid, y_valid))  # Fit model

