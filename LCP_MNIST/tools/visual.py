import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.ney import CNN

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
model = CNN().to(device)
model.load_state_dict(torch.load('../models/mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载和预处理图像
def predict_and_display(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)

    predicted_label = predicted.item()

    # 显示图像和预测结果
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()

# 示例使用
image_path = '../data/shouxieshuzi100_plus/augmented_0.jpg'  # 替换为你要预测的图像路径
predict_and_display(image_path)
