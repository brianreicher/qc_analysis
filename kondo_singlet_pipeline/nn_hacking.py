import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

X=np.random.random( size=(100,1) )
y=np.random.randint(0,100,size=(100,3)).astype(float)   #Regression

input1 = Input(shape=(1,))
l1 = Dense(10, activation='relu')(input1)
l2 = Dense(50, activation='tanh')(l1)
l3 = Dense(50, activation='relu')(l2)
out = Dense(3)(l3)

model = Model(inputs=input1, outputs=[out])
model.compile(
    optimizer='adam',
    loss=['mean_squared_error']
    )
history = model.fit(X, [y], epochs=10, batch_size=100)


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(100, 1, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")