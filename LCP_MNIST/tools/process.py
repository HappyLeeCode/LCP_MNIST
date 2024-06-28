import os
import cv2

# 设置输入和输出文件夹
input_folder = 'G:/abcd'
output_folder = 'G:/abcde'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .jpg 图像文件
for filename in os.listdir(input_folder):
    if filename.endswith( '.jpg'):
        # 读取图像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # 第一次膨胀
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_dilated1 = cv2.dilate(img, kernel1, iterations=1)
        
        # 第二次膨胀
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        img_dilated2 = cv2.dilate(img_dilated1, kernel2, iterations=1)
        
        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_dilated2)

print('图片处理完成!')