import os
import cv2
import pickle # 用于保存变量
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # 展示进度条
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator # 用于增强数据
from keras import regularizers
import tensorflow as tf

np.random.seed(22)

'''
加载数据集
'''

# 用于加载正常图片并归一化的函数
def load_normal(norm_path):
    norm_files = np.array(os.listdir(norm_path))
    norm_labels = np.array(['normal']*len(norm_files))

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

    return norm_images, norm_labels

# 用于加载测肺炎图片并归一化的函数
def load_pneumonia(pneu_path):
    pneu_files = np.array(os.listdir(pneu_path))
    pneu_labels = np.array(['pneumonia']*len(pneu_files))

    pneu_images = []
    for image in tqdm(pneu_files):
        # 读取图片
        image = cv2.imread(pneu_path + image)
        # 归一化，重置大小为200*200
        image = cv2.resize(image, dsize=(200,200))
        # 转化为灰度图（单通道）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pneu_images.append(image)

    pneu_images = np.array(pneu_images)

    return pneu_images, pneu_labels

# 加载训练集
print('Loading images')
norm_images, norm_labels = load_normal('E:/All_Project/grade3/AI_tech/CNN2/chest_xray/train/NORMAL/')
pneu_images, pneu_labels = load_pneumonia('E:/All_Project/grade3/AI_tech/CNN2/chest_xray/train/PNEUMONIA/')

# 将数组中图片和标签分别储存为x、y
X_train = np.append(norm_images, pneu_images, axis=0)
y_train = np.append(norm_labels, pneu_labels)

print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

# 显示部分训练集图片
print('Display several images')
fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(16, 4))

indices = np.random.choice(len(X_train), 14)
counter = 0

for i in range(2):
    for j in range(7):
        axes[i,j].set_title(y_train[indices[counter]])
        axes[i,j].imshow(X_train[indices[counter]], cmap='gray')
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        counter += 1
plt.show()

# 加载验证集
print('Loading test images')
norm_images_test, norm_labels_test = load_normal('E:/All_Project/grade3/AI_tech/CNN2/chest_xray/val/NORMAL/')
pneu_images_test, pneu_labels_test = load_pneumonia('E:/All_Project/grade3\AI_tech/CNN2/chest_xray/val/PNEUMONIA/')
X_test = np.append(norm_images_test, pneu_images_test, axis=0)
y_test = np.append(norm_labels_test, pneu_labels_test)

# 保存已加载的图片
with open('pneumonia_data.pickle', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
with open('pneumonia_data.pickle', 'rb') as f:
    (X_train, X_test, y_train, y_test) = pickle.load(f)

'''
标签预处理
'''

print('Label preprocessing')
# Create new axis on all y data
y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]

# 初始化 OneHotEncoder 对象
one_hot_encoder = OneHotEncoder(sparse=False)

# 将标签数据转换为 one-hot
y_train_one_hot = one_hot_encoder.fit_transform(y_train)
y_test_one_hot = one_hot_encoder.transform(y_test)

'''
重塑数据
'''

print('Reshaping X data')
# 把数据重塑为(none, height, width, 1),1 代表单个颜色通道
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

'''
数据扩充
'''

print('Data augmentation')

datagen = ImageDataGenerator(
        rotation_range = 10,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip=True)

datagen.fit(X_train)
train_gen = datagen.flow(X_train, y_train_one_hot, batch_size = 32)

'''
搭建CNN网络
'''

print('CNN')
# 定义输入形状
input_shape = (X_train.shape[1], X_train.shape[2], 1)
print(input_shape)

# 输入层
input1 = Input(shape=input_shape)

'''
# （初代网络）非正则化
# 卷积1
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(input1)
# 卷积2（正则化）
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
# 池化
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)



# 初代网络+正则化
# 卷积1（正则化）
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(input1)
# 卷积2（正则化）
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
# 池化
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = MaxPool2D((2, 2))(cnn)

cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same', kernel_regularizer=regularizers.l2(0.01))(cnn)
cnn = MaxPool2D((2, 2))(cnn)
'''


# 进阶网络（无正则化）
# 第1组
# 卷积1
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(input1)
# 卷积2（正则化）
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
# 池化
cnn = MaxPool2D((2, 2))(cnn)

# 第2组
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

# 第3组
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

# 第4组
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

# 第5组
cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

# 第6组
cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)

# 第7组
cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
    padding ='same')(cnn)
cnn = MaxPool2D((2, 2))(cnn)



# 全连接
cnn = Flatten()(cnn)
cnn = Dense(100, activation='relu')(cnn)
cnn = Dense(50, activation='relu')(cnn)

# 输出层
output1 = Dense(2, activation='softmax')(cnn)

model = Model(inputs=input1, outputs=output1)


'''
编译神经网络
'''
# 设置损失函数，设置优化器（sgd）
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['acc'])

'''
训练模型
'''
history = model.fit_generator(train_gen, epochs=60,
          validation_data=(X_test, y_test_one_hot))

# 保存模型
model.save('pneumonia_cnn.h5')

'''
显示模型性能
'''
# 准确度
print('Displaying accuracy')
plt.figure(figsize=(8,6))
plt.title('Accuracy scores')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.show()

# 损失值
print('Displaying loss')
plt.figure(figsize=(8,6))
plt.title('Loss value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

'''
测试
'''
# 测试数据
predictions = model.predict(X_test)
print(predictions)

predictions = one_hot_encoder.inverse_transform(predictions)

print('Model evaluation')
print(one_hot_encoder.categories_)

classnames = ['normal','pneumonia']

# 绘制混淆矩阵
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8,8))
plt.title('Confusion matrix')
sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()