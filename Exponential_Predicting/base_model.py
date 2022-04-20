import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

X=np.random.random(size=(100,1))
y=np.random.randint(0,100,size=(100,3)).astype(float)   #Regression

input1 = Input(shape=(1,))
l1 = Dense(10, activation='relu')(input1)
l2 = Dense(50, activation='relu')(l1)
l3 = Dense(50, activation='relu')(l2)
out = Dense(3)(l3)

model = Model(inputs=input1, outputs=[out])
model.compile(
    optimizer='adam',
    loss=['mean_squared_error']
    )

history = model.fit(X, [y], epochs=10, batch_size=64)