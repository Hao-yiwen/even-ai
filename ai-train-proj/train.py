import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 定义超参数和设备
EPOCHS = 5  # 训练轮次，可以设为 3-5 快速查看效果
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 检查是否有可用的 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 加载和预处理 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',        # 数据集下载路径
    train=True,           # 是训练集
    transform=transform,  # 应用转换
    download=True         # 如果本地没有，则下载
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,          # 是测试集
    transform=transform,
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 定义简单的卷积神经网络 (CNN) 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # 输入: (Batch, 1, 28, 28), 输出: (Batch, 16, 28, 28)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输出: (Batch, 16, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 输出: (Batch, 32, 14, 14)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输出: (Batch, 32, 7, 7)

        # 全连接层
        # 32 (channels) * 7 (height) * 7 (width) = 1568
        self.fc = nn.Linear(32 * 7 * 7, 10) # 10个类别 (数字0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # 展平操作
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 训练模型
train_losses = []
for epoch in range(EPOCHS):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 将数据移动到指定的设备 (GPU 或 CPU)
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad() # 清空梯度
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新权重

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0: # 每100个batch打印一次信息
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {epoch_loss:.4f}')

print('Finished Training')

# 6. (可选) 保存模型
# torch.save(model.state_dict(), 'mnist_cnn_model.pth')
# print('Model saved to mnist_cnn_model.pth')

# 7. 评估模型
model.eval() # 设置模型为评估模式
with torch.no_grad(): # 在评估阶段不计算梯度
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {total} test images: {accuracy:.2f} %')

# 8. (可选) 可视化训练损失
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()