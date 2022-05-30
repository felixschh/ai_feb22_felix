import data_handler as dh        
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import Network

net = Network(8)

x_train, x_test, y_train, y_test = dh.load_data('Chapter 03/03. MLP Regression/data/turkish_stocks.csv')

optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 50

train = []
test = []

for epoch in range(epochs):
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = dh.to_batches(x_train, x_test, y_train, y_test, 8)
    running_loss = 0
    running_loss_test = 0

    for n in range(x_train_batch.shape[0]):
        optimizer.zero_grad()
        pred = net.forward(x_train_batch[n])
        loss = criterion(pred, y_train_batch[n])

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_pred = net.forward(x_test_batch)
            test_loss = criterion(test_pred, y_test_batch)

        running_loss += loss.item()
        running_loss_test += test_loss.item()
    train.append(running_loss/len(x_train_batch))
    test.append(running_loss_test/len(x_test_batch))
    print(f'{epoch + 1}/ {epochs}')


plt.plot(train, label='train Loss')
plt.plot(test, label='test Loss')

plt.legend()
plt.show()