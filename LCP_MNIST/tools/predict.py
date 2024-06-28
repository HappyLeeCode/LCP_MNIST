import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
#from models.ney import CNN
from models.vgg16 import VGG16
from models.LeNet import Module
from models.OurModules.best_ney import CNN
import config

# 设置设备
device = config.device

# 加载和预处理数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
model = CNN().to(device)
# model.load_state_dict 加载之前训练好的模型参数。如果使用GPU训练，则需要把模型转为CPU模式
model.load_state_dict(torch.load(config.model_save_path, map_location=torch.device('cpu')))
model.eval()

#  初始化 correct 和 total 变量来分别存储正确预测的数量和总的测试样本数量。
correct = 0
total = 0
# 使用 torch.no_grad() 上下文管理器来禁用梯度计算，
# 这在测试时是必要的，以减少内存使用并加快速度。
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # 遍历 test_loader中的批次数据，将数据移动到设备上，
        # 通过模型进行前向传播，使用torch.max确定预测的类别。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
