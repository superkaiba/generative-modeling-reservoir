import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import argparse
import os

class Net(nn.Module):
    def __init__(self, num_layers, width, c):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(width, width, bias=True)
            layer.weight.data.uniform_(-c, c)
            layer.weight.requires_grad = False
            self.layers.append(layer)
        self.output_layer = nn.Linear(width, 10, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

def main(c, num_layers, width, logdir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(trainset))
    test_size = len(trainset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(trainset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    net = Net(num_layers, width, c)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.view(inputs.shape[0], -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.view(images.shape[0], -1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {correct / total}')
        with open(os.path.join(logdir, 'log.csv'), 'a') as f:
            f.write(f'{epoch + 1},{running_loss / len(trainloader)},{correct / total}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a feedforward neural network on MNIST')
    parser.add_argument('--c', type=float, required=True, help='the max of the uniform distribution')
    parser.add_argument('--num_layers', type=int, required=True, help='the number of hidden layers in the neural network')
    parser.add_argument('--width', type=int, required=True, help='the width of each layer in the network')
    parser.add_argument('--logdir', type=str, required=True, help='where to log the results')
    args = parser.parse_args()
    main(args.c, args.num_layers, args.width, args.logdir)
