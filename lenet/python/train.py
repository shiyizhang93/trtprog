import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from model import LeNet



DIR = './weights'
WTS = 'mnist_net.pt'


def trans_img():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    return transform


def train_dataloader(transform, batch_size):
    trainset = datasets.MNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    return trainloader


def train(batch_size, epochs, learning_rate, momentum, transform, net, device):
    trainloader = train_dataloader(transform, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    path = os.path.join(DIR, WTS)
    torch.save(net.state_dict(), path)
    print('Saved training weights')


if __name__ == "__main__":
    batch_size = 4
    epochs = 4
    learning_rate = 0.001
    momentum = 0.9
    transform = trans_img()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lenet = LeNet().to(device)
    train(batch_size, epochs, learning_rate, momentum, transform, lenet, device)

