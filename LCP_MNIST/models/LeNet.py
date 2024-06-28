import torch
from torch import nn


# 搭建神经网络
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # 初始化模型
        self.model = nn.Sequential(
            # 卷积层，输入通道数为3，输出通道数为16，卷积核大小为5
            nn.Conv2d(1, 16, 5),
            # 最大池化层，池化窗口大小为2x2
            nn.MaxPool2d(2, 2),
            # 卷积层，输入通道数为16，输出通道数为32，卷积核大小为5
            nn.Conv2d(16, 32, 5),
            # 最大池化层，池化窗口大小为2x2
            nn.MaxPool2d(2, 2),
            # 展平层，将多维的输入一维化，用于输入到全连接层
            # 注意一下,线性层需要进行展平处理
            nn.Flatten(), 
            # 全连接层，输入节点数为32*5*5（根据前面卷积和池化后的特征图大小计算得出），输出节点数为120
            nn.Linear(32 * 4 * 4, 120),
            # 全连接层，输入节点数为120，输出节点数为84
            nn.Linear(120, 84),
            # 全连接层，输入节点数为84，输出节点数为10（假设为分类问题的类别数）
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # 将输入x传入模型进行前向传播
        x = self.model(x)
        return x
