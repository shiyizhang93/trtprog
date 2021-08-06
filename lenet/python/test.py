import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import LeNet


PATH = './weights/mnist_net.pt'
# PATH = './weights/mnist_net_int8.pt'

def trans_img():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    return transform


def test_dataloader(transform, batch_size):
    testset = datasets.MNIST(root='./data',
                             train=False,
                            download=True,
                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)
    return testloader


def test(net, testloader, device):
    correct = 0
    total = 0
    net.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lenet = LeNet().to(device)
    batch_size = 4
    transform = trans_img()
    testloader = test_dataloader(transform, batch_size)
    test(lenet, testloader, device)
