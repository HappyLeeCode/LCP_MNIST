import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from models.ney import CNN
from models.vgg16 import VGG16
from models.OurModules.best_ney import CNN
from tensorboardX import SummaryWriter
import config
from models.LeNet import  Module

# 设置设备
device = config.device

# 加载和预处理数据
'''
transforms.Compose(): 这是一个容器，它将多个变换函数组合成一个列表。

transforms.ToTensor(): 这个变换将图像数据转换为 torch.FloatTensor 类型，并将数值范围从 [0, 255] 转换为 [0.0, 1.0]。

transforms.Normalize(mean, std): 这个变换用于将图像数据标准化，使其具有指定的均值 mean 和标准差 std。
在这个例子中，(0.5,) 表示均值为 0.5，(0.5,) 表示标准差也为 0.5。
这意味着每个通道的像素值将被减去 0.5 并除以 0.5，从而将数据转换到 [-1, 1] 的范围内。
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载批大小 64  /32
batch_size = config.batch_size
# 加载训练集和验证集路径
train_path = config.train_path
val_path = config.var_path
train_dataset = torchvision.datasets.MNIST(root=train_path, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torchvision.datasets.MNIST(root=val_path, train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型

model = CNN().to(device)
#model = Module().to(device)
#model = VGG16().to(device)
# 设置损失函数和优化器

criterion = nn.CrossEntropyLoss()
# 设置学习率 学习率下降策略
lr = config.lr
# 设置动量
momentum = config.momentum
# 设置优化算法
#optimizer = optim.Adam(model.parameters(), lr, momentum)
optimizer = optim.Adam(model.parameters(), lr)

# 配置 TensorBoard
log_dir = "../logs/fit_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# 训练模型
# 选择模型训练总步数
num_epochs = config.num_epochs
# 记录当前训练步数
global_step = config.global_step

# 开始循环训练
for epoch in range(num_epochs):
    # 模型设置为训练模式
    model.train()
    # 初始化一个变量来累计批次损失，以便计算平均损失。
    running_loss = 0.0
    # 循环遍历训练数据加载器 train_loader 中的批次，
    # enumerate 函数返回批次索引 i 和数据 (images, labels)
    for i, (images, labels) in enumerate(train_loader):
        # 将图像和标签数据移动到指定的设备（如 GPU）上
        images, labels = images.to(device), labels.to(device)
        # 在反向传播之前清除之前的梯度，因为 PyTorch 会累积梯度。
        optimizer.zero_grad()
        # 将当前批次的图像数据通过模型前向传播，得到输出。
        outputs = model(images)
        # 使用损失函数（如交叉熵损失）计算模型输出和真实标签之间的损失。
        loss = criterion(outputs, labels)
        # 计算损失相对于模型参数的梯度
        loss.backward()
        # 使用优化器（如 SGD 或 Adam）根据计算出的梯度更新模型的参数
        optimizer.step()
        # 将当前批次的损失添加到累计损失中。
        running_loss += loss.item()
        # 每 100 个批次打印一次平均损失。这是通过检查批次索引 i 除以 100 的余数是否为 99 来实现的
        if i % 100 == 99:
            # 计算100个批次的平均损失。
            avg_loss = running_loss / 100
            # 在打印完平均损失后，重置累计损失为 0，以便计算下一个 100 个批次的平均损失
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

            # 记录训练损失到 TensorBoard， training loss' 是标签，avg_loss 是损失值，global_step 是全局步数，通常用于记录训练进度
            writer.add_scalar('training loss', avg_loss, global_step)
            # 全局步数递增: global_step += 1 表示每次记录后全局步数递增。
            global_step += 1

    # 在每个 epoch 结束后计算验证损失
    #  将模型设置为评估模式，这会关闭 Dropout 和 Batch Normalization 等层的训练行为
    model.eval()
    # 计算验证损失: 初始化 val_loss 为 0，
    # 然后遍历验证数据加载器 val_loader 中的数据，计算损失并将损失值累加。
    val_loss = 0.0
    # 这个上下文会告诉 PyTorch 不需要计算梯度，这在评估模型时很有用，因为它减少了内存消耗并加快了计算速度。
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # 计算整个验证集的平均验证损失
    avg_val_loss = val_loss / len(val_loader)

    # 记录验证损失到 TensorBoard
    writer.add_scalar('validation loss', avg_val_loss, epoch)
    # tensorboard --logdir="logs/fit/"
    # 将训练和验证损失同时打印出来
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

print('Finished Training')
# 保存模型:
# 使用 torch.save 函数将模型的参数（state_dict）保存到文件 './models/mnist_cnn.pth'。
# 这样，训练好的模型可以在以后重新加载和使用
model_save_path = config.model_save_path
torch.save(model.state_dict(), model_save_path)
