# import torch
# import numpy as np
# from generate_data import data
# from torch import nn
# from torch.utils.data import DataLoader
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
#
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(len(data.columns), 100),
#             nn.ReLU(),
#             nn.Linear(100, 100),
#             nn.ReLU(),
#             nn.Linear(30, 3),
#         )
#
#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
#
# model = NeuralNetwork().to(device)
#
# #Drop indicies for labels
# drop_vals = np.arange(len(data.columns)-1).tolist()
# for i in drop_vals:
#     drop_vals[i] = str(drop_vals[i])
#
# training_data = data.drop(columns = "labels").ToTensor()
# test_data = data.drop(columns= drop_vals).ToTensor()
#
# train_dataloader = DataLoader(training_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)
#
# learning_rate = 1e-3
# batch_size = 64
# epochs = 5
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")