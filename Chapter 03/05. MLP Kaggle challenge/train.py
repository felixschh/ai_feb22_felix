from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, no_grad
from torch import optim
import torch.nn.functional as F
import data_handler as dh
from model import Classify


trainset, testset, trainloader, testloader = dh.get_data()

model = Classify(784)

criterion =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 30
train = []
test = []
print_every = 40
starting_time = time.time()

for epoch in range(epochs):
    running_loss_train = 0
    running_loss_test = 0

    if epoch%10 == 0:
        torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')

    print(f'Epoch: {epoch+1}/{epochs}')
    
    for i, (images, labels) in enumerate(iter(trainloader)):

        images.resize_(images.size()[0], 784)
        optimizer.zero_grad()
        logits = model.forward(images)
        train_loss = criterion(logits, labels)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(iter(testloader)):
                images.resize_(images.size()[0], 784)
                logits = model.forward(images)
                test_loss = criterion(logits, labels)
  
        model.train()

        running_loss_train += train_loss.item()
        running_loss_test += test_loss.item() 

        if i%print_every ==0:
                print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss=0

    train.append((running_loss_train/len(trainloader)))
    test.append((running_loss_test/len(testloader)))

# train_time = time.time() - starting_time
# print(f'Time: {train_time}')

plt.plot(train, label = 'Trainloss')
plt.plot(test, label = 'Testloss')
torch.save(model.state_dict(), 'checkpoint.pth')

plt.plot()
plt.legend()
plt.show()
