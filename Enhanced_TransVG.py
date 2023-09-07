"""
@ author: neo
@ date: 2023-09-07  11:06 
@ file_name: 1.PY
@ github: https://github.com/Underson888/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据加载器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


# 第一种模型
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(3072 * 2, 1024)
        self.attn = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x], dim=1)
        x = F.relu(self.fc1(x))
        attn_weights = F.softmax(self.attn(x), dim=1)
        x = torch.mul(x, attn_weights)
        x = self.fc2(x)
        return x


# 第二种模型
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.memory = nn.Parameter(torch.rand(3072))
        self.fc1 = nn.Linear(3072 * 3, 1024)
        self.attn = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x, self.memory.repeat(x.size(0), 1)], dim=1)
        x = F.relu(self.fc1(x))
        attn_weights = F.softmax(self.attn(x), dim=1)
        x = torch.mul(x, attn_weights)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = Model1().to(device)
model2 = Model2().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=0.0005)
optimizer2 = optim.Adam(model2.parameters(), lr=0.0005)

num_epochs = 30

train_loss_list1 = []
test_loss_list1 = []
test_acc_list1 = []

train_loss_list2 = []
test_loss_list2 = []
test_acc_list2 = []

for epoch in range(1, num_epochs + 1):
    print("\nModel 1, Epoch:", epoch)
    train_loss1 = train(model1, device, train_loader, optimizer1, epoch)
    test_loss1, test_acc1 = test(model1, device, test_loader)
    train_loss_list1.append(train_loss1)
    test_loss_list1.append(test_loss1)
    test_acc_list1.append(test_acc1)

    print("\nModel 2, Epoch:", epoch)
    train_loss2 = train(model2, device, train_loader, optimizer2, epoch)
    test_loss2, test_acc2 = test(model2, device, test_loader)
    train_loss_list2.append(train_loss2)
    test_loss_list2.append(test_loss2)
    test_acc_list2.append(test_acc2)

# 绘制Loss曲线
plt.figure()
plt.plot(train_loss_list1, 'b', label='Model1 Loss')
# plt.plot(test_loss_list1, 'r', label='Model1 Test Loss')
plt.plot(train_loss_list2, 'c', label='Model2 Loss')
# plt.plot(test_loss_list2, 'm', label='Model2 Test Loss')
plt.title('Loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制Accuracy曲线
plt.figure()
plt.plot(test_acc_list1, 'b', label='Model1 Accuracy')
plt.plot(test_acc_list2, 'r', label='Model2 Accuracy')
plt.title('Accuracy curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
