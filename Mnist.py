import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.onnx as onnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_data=datasets.MNIST(root='/home/cy/Mnist/data',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='/home/cy/Mnist/data',train=False,download=True,transform=transform)
trainloder=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=2)
testloder=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=2)

#构建神经网络
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)#卷积层，处理图像的二维数据
        self.relu1 = nn.ReLU()#激活函数
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)#最大池化层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()#将多维数据展成一维
        self.linear1 = nn.Linear(7*7*64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.linear1(x))
        x = self.linear2(x)
        return x


model=MnistCNN().to(device)#读取命令并在设备上运行
criterion = nn.CrossEntropyLoss()#损失函数
criterion=criterion.to(device)
learningrate=0.01#设置学习率
optimizer=torch.optim.Adam(model.parameters(),lr=learningrate)

epoch=10
for epoch in range(epoch):
    running_loss=0.0
    model.train()
    for i, data in enumerate(trainloder,0):
        inputs,labels=data
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()#优化梯度置为0
        loss.backward()#反向传播
        optimizer.step()#更新参数
        running_loss += loss.item()
        
        if i%200 ==199:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
model.eval()
print("finish")
#训练前的初始值
correct=0
samples=0

with torch.no_grad():
    for data in testloder:
        images,labels=data
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)#计算损失值
        _, predictions = torch.max(outputs.data, dim=1)
        samples+=labels.size(0)
        correct += (predictions == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / samples}%')

dummy_input = torch.randn(1, 1, 28, 28).to(device)

onnx.export(model, dummy_input, 'mnist_model.onnx')
