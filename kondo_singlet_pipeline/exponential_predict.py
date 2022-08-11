import numpy as np
import pandas as pd
import torch
import tensorflow
from torch import nn
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Returns a dataframe of calculated exponential functions for given C-parameters and L-angular quantum numbers
class GenerateData:
    def __init__(self, l):
        self.l = l
        self.x = np.arange(l)
        self.c = np.arange(start=0.0001, stop=1, step=0.0001)

    # Evaluate function for given c-values
    @staticmethod
    def func(input, arg):
        return np.e ** (-input / arg)

    # Create dataset: 100,000 funcs?
    def populate(self):
        func_df = pd.DataFrame(index=self.c.tolist())
        for i in self.x:
            col = []
            for j in self.c:
                col.append(GenerateData(self.l).func(input=i, arg=j))
            func_df[i] = col
        func_df['c'] = self.c
        return func_df

    # Set param value for integer L-value (angular momentum quantum number)
    def Data(self):
        return GenerateData(self.l).populate()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(data.columns), 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(30, 3),
        )

    def forward(self, x):
        forward = self.linear_relu_stack(x)
        return forward




if __name__ == '__main__':
    data = GenerateData(4).Data()
    print(data)


    model = NeuralNetwork()
    print(model)
    # Drop indicies for labels
    drop_vals = np.arange(len(data.columns) - 1).tolist()

    # Set training and testing datasets and dataloaders
    training_data = test_data = data.to_numpy()

    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=1)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(2):  # loop over the dataset multiple times
        count = 0
        running_loss = 0.0
        for data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            labels = data[0][len(data - 1)]

            l = len(data[0]) - 1
            inputs = data[0][0:l]
            count += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training')

        X = torch.rand(100, 1, device=device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        print(f"Predicted class: {y_pred}")