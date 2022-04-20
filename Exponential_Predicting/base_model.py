import numpy as np
import tensorflow
import keras
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
    model.add(Dense(10, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(7))
    return model


model = load_model() #Initiate model
keras.backend.clear_session() #Clear model backend and set random state
np.random.seed(42)
tensorflow.random.set_seed(42)

model.compile(optimizer='Adam', #Complile model
              loss=['mean_squared_error', 'huber_loss'],
              metrics=['cosine_similarity', 'hinge'])

history = model.fit(x_train, [y_train], epochs=60, batch_size=20,
                    validation_data=(x_valid, y_valid)) #Fit model
