import torch
import torchvision.transforms as transforms  # 图像预处理库
from PIL import Image  # 用于图像处理
#from models.ney import CNN
from models.OurModules.best_ney import CNN
from models.vgg16 import VGG16
from models.LeNet import Module
import os  # 用于文件和目录操作的标准库。
import config

# 设置设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = config.device

# 定义模型
'''
实例化 CNN 模型并将其移动到定义的设备上。
加载预训练的模型参数。
将模型设置为评估模式
'''
model = CNN().to(device)
model.load_state_dict(torch.load(config.model_save_path, map_location=torch.device('cpu')))
model.eval()

# 定义预处理
'''
包括灰度化、调整大小(28*28)、转换为张量和标准化
'''
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(config.input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 加载和预处理图像
# 接收图像路径，打开并预处理图像
def predict(image_path):
    image = Image.open(image_path)
    # unsqueeze(0)在每张图片之前增加一个维度
    image = transform(image).unsqueeze(0).to(device)

    # 预测:
    # 使用模型进行预测并返回最大概率的类别
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()


# 修改 evaluate_folder 函数以处理整个文件夹，并计算准确率
def evaluate_folder(folder_path):
    total_images = 0
    correct_predictions = 0
    results = []

    # 遍历文件夹中的子文件夹
    for label in range(10):
        '''
        使用 os.path.join 函数来拼接路径，创建每个类别的文件夹路径。
        folder_path 是包含所有类别子文件夹的主文件夹路径，
        label 是当前类别的索引，转换为字符串后作为子文件夹的名称。
        '''
        label_folder = os.path.join(folder_path, str(label))
        if not os.path.isdir(label_folder):
            continue

        image_files = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]

        # 遍历图像文件
        for image_file in image_files:
            image_path = os.path.join(label_folder, image_file)

            # 预测图像标签
            predicted_label = predict(image_path)

            # 统计正确和错误的预测
            if predicted_label == label:
                correct_predictions += 1
            total_images += 1

            # 保存结果
            results.append(f'{image_path}\t{label}\t{predicted_label}\n')

    # 计算准确率
    accuracy = (correct_predictions / total_images) * 100

    # 将结果写入文件
    with open('../prediction_results.txt', 'w') as f:
        f.writelines(results)

    return accuracy


if __name__ == "__main__":
    # 示例使用
    folder_path = '../data/test_data'  # 替换为你的文件夹路径
    accuracy = evaluate_folder(folder_path)
    # 打印准确率
    print(f'示例测试集总准确率：{accuracy:.2f}%')
    # 测试使用
    folder_path = '../our_dataset/test_data'  # 替换为你的文件夹路径
    accuracy = evaluate_folder(folder_path)
    # 打印准确率
    print(f'自制测试集总准确率：{accuracy:.2f}%')
