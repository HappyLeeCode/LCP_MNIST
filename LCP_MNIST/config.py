
# 训练参数配置
device = 'cpu'
train_path = '../data'
var_path = '../data'
test_path = ''

model_save_path = '../models/mnist_cnn.pth'
#model_save_path = '../models/mnist_cnn_origin.pth'
#model_save_path = '../models/mnist_module_origin.pth'
#model_save_path = '../models/mnist_vgg_origin.pth'

input_size = (28, 28)
batch_size = 64  # 批大小
num_epochs = 10  # 训练步数
global_step = 0  # 当前步数起点
lr = 0.001  # 学习率
momentum = 0.91  # 优化器动量
