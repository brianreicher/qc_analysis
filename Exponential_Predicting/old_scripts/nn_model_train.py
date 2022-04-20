from operator import mod
from re import X
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from generate_data import data


# - Consider the exponential function f(x)=exp(-x/c), with x=0,1,2,….,L-1 integers
# - Generate many training datasets {f(0,f(1),f(2),…f(L-1)} for different values of c
# - Now we solve the inverse problem: input c and see if you get the exponential.

# Initate Dataloader - pick l-value of choice
dataloader = data(7)
drop_vals = np.arange(len(dataloader.columns)-1).tolist()
x_train, x_valid, y_train, y_valid = train_test_split(dataloader.drop(columns='c').to_numpy(), 
                                                        dataloader.drop(columns = drop_vals).to_numpy(), 
                                                        test_size=0.2, random_state=0)

# Initate model
def load_hyper_param_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, activation="relu"))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(7))
    # print(model.summary())
    # keras.utils.plot_model(model, "model_graph.png", show_shapes=True)
    return model

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = load_hyper_param_model()

keras.backend.clear_session()
np.random.seed(42)
tensorflow.random.set_seed(42)
model.compile(loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=[rmse])
history = model.fit(x_train, y_train,
        batch_size=30, epochs=30,
        verbose=1)


print(model.evaluate(x_valid, y_valid))