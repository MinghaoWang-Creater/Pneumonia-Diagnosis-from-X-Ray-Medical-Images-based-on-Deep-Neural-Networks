import os
from keras.models import load_model
import numpy as np
import cv2
from tqdm import tqdm

# 加载模型文件
model = load_model("pneumonia_cnn.h5")
# model.summary() #输出模型各层的参数状况

# 归一化
def get_inputs(norm_path):
    norm_files = np.array(os.listdir(norm_path))

    norm_images = []
    for image in tqdm(norm_files):
        # 读取图片
        image = cv2.imread(norm_path + image)
        # 归一化，重置大小为200*200
        image = cv2.resize(image, dsize=(200,200))
        # 转化为灰度图（单通道）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        norm_images.append(image)

    norm_images = np.array(norm_images)

    return norm_images

# 读取测试数据，并归一化
pre_x = get_inputs('E:/All_Project/grade3/AI_tech/CNN2/test/')

# 预测
pre_y = model.predict(pre_x)
# print(pre_y)
print(np.argmax(model.predict(pre_x),axis=1))
