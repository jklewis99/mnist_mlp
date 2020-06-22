from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

"""## CNN"""


class MNISTImageDataset(Dataset):
    def __init__(self, x, y):
        self.x = x / 255
        self.y = y

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        x = x.view((1, 28, 28))
        y = torch.tensor(self.y[idx])
        return x, y

    def __len__(self):
        return self.x.shape[0]


# Instantiating the MNISTDataset
trainset = MNISTImageDataset(images_train, labels_train)
testset = MNISTImageDataset(images_test, labels_test)

# Visualizing a data sample

image, label = trainset[0]

plt.imshow(image.reshape((28, 28)))
print(label)

# Creates an iterable object that iterates over batches of specific batch size.
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
testloader = DataLoader(testset, batch_size=100)


class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, (3, 3))  # 26 26
        self.pool = nn.MaxPool2d(2, 2)  # 13 13
        self.conv2 = nn.Conv2d(16, 16, (3, 3))  # 11 11
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x


model = CNN(1, 10)
model.cuda()


criterion = nn.CrossEntropyLoss()
criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 100
print_every = 100
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # move the data to GPU
        inputs = inputs.cuda().float()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_acc += (torch.sum(outputs.argmax(dim=1)
                                  == labels).item() / 100)

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every-1:
            print('[%d, %5d] loss: %.3f - acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_every, running_acc * 100 / print_every))
            running_loss = 0.0
            running_acc = 0.0

print('Finished Training')


df = pd.read_csv('mnist_train_small.csv', header=None)
df

data = df.to_numpy()

data.shape

images = data[:, 1:]
labels = data[:, 0]

images.shape, labels.shape

images = images.reshape((-1, 28, 28))
images.shape

idx = 1000
plt.imshow(images[idx])
print(labels[idx])


# define transforms
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0, ), (1, ))
     ])


class MNISTData(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.transform(self.X[idx].astype(np.uint8))
        y = self.Y[idx]
        return x, y


data = MNISTData(images, labels, transform)

dataloader = DataLoader(data, batch_size=100, shuffle=True)

for x in dataloader:
    print(x)
    break
