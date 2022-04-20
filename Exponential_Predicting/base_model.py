import numpy as np
import tensorflow
import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from generate_data import data


# Regression testing
l_val = 7
dataloader = data(l_val)
drop_vals = np.arange(len(dataloader.columns)-1).tolist()
x_train, x_valid, y_train, y_valid = train_test_split(dataloader.drop(columns = drop_vals).to_numpy(), 
                                                      dataloader.drop(columns='c').to_numpy(), test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(1, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(7))

keras.backend.clear_session()
np.random.seed(42)
tensorflow.random.set_seed(42)
model.compile(
    optimizer='adam',
    loss=['mean_squared_error']
    )

history = model.fit(x_train, [y_train], epochs=10, batch_size=64)