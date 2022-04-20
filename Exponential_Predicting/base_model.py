import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split
from generate_data import data


# Regression testing
l_val = 7
dataloader = data(l_val)
drop_vals = np.arange(len(dataloader.columns)-1).tolist()
x_train, x_valid, y_train, y_valid = train_test_split(dataloader.drop(columns = drop_vals).to_numpy(), 
                                                      dataloader.drop(columns='c').to_numpy(), test_size=0.2, random_state=0)
input1 = Input(shape=(1,))
l1 = Dense(10, activation='relu')(input1)
l2 = Dense(50, activation='relu')(l1)
l3 = Dense(50, activation='relu')(l2)
out = Dense(l_val)(l3)

model = Model(inputs=input1, outputs=[out])
model.compile(
    optimizer='adam',
    loss=['mean_squared_error']
    )

history = model.fit(x_train, [y_train], epochs=10, batch_size=64)