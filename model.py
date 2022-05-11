import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# Make pytorch run on GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


# Get data
traintransform = transforms.Compose([transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(size=32, padding=4),
#    transforms.RandomVerticalFlip(),
#    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#    transforms.RandomRotation(10, expand=True),
#    transforms.Resize((32, 32)),
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
testtransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 50

trainset= torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=traintransform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testtransform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# The classifications
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
Structure of a ResNet
1 : A convolution block
2 : ResLayer 1 (3 blocks)
3 : ResLayer 2 (4 blocks)
4 : ResLayer 3 (7 blocks)
5 : ResLayer 4 (3 blocks)
6 : A classifer block
'''

# Define the model
'''Creating ResBlocks'''
class ResBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, downsize=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsize = downsize

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #If original size does not match current one, then downsize
        if self.downsize: out += self.downsize(x)
        else: out += x

        return F.relu(out)

'''Creating the ResNet class (Putting it all together)'''
class ResLayers(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.l1 = self.MakeLayer(block, 64, layers[0], stride=1)
        self.l2 = self.MakeLayer(block, 128, layers[1], stride=2)
        self.l3 = self.MakeLayer(block, 256, layers[2], stride=2)
        self.l4 = self.MakeLayer(block, 512, layers[3], stride=2)
        self.fully_connected = nn.Linear(512, num_classes)
    
    '''Creating the ResLayer'''
    def MakeLayer(self, block, planes, blocks, stride=1):
        downsize = None
        if stride != 1 or self.in_planes != planes:
            downsize = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(self.in_planes, planes, stride, downsize)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
#        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
#        print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
#        print(x.shape)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
#        print(x.shape)
        x = self.fully_connected(x)
#        print(x.shape)

        return x

'''Make Model'''
def ResNet():
    layers = [3,4,7,3]
    model = ResLayers(ResBlock, layers)
    #model.load_state_dict(torch.load(os.getcwd()+'/ResNetCifar10.txt'))
    return model

model = ResNet()

# Train the model
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def saveModel():
    torch.save(model.state_dict(), os.getcwd()+'/ResNetCifar10.txt')

def test():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader, 0):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted==labels).sum().item()

    model.train()
    return (100 * accuracy / total)

def train(num_epochs):
    best_accuracy = 0.0
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        current_loss = 0.0
        acc = 0.0

        for i, (images, labels) in enumerate(trainloader, 0): 
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            if i%100==99:
                print(f'({epoch+1}, {i+1}) loss -> {round(current_loss/1000, 3)}')
                current_loss = 0

        acc = test()
        output = f'For epoch {epoch+1} the test accuracy over the whole test set is {acc} %'
        print(output)

        #storing results
        file = open('t1.txt', 'a')
        file.write(f'Epoch #{epoch+1} -> {acc}\n')
        file.close()

        if acc > best_accuracy:
            saveModel()
            best_accuracy = acc

train(1000)
