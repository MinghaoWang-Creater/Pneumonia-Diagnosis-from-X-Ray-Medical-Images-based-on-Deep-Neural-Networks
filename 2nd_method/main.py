from tensorflow.python import keras
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import itertools
import os

'''
tensorflow与cuDNN版本不对应时可使用
'''
'''
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''


'''
初始设置
'''
# 参数设置
im_height = 200
im_width = 200
batch_size = 8
epochs = 10

# 模型保存路径设置
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

# 数据集路径
image_path = "E:/All_Project/grade3/AI_tech/CNN1/chest_xray/"
train_dir = image_path + "train"
validation_dir = image_path + "test"
test_dir = image_path + "val"


'''
数据预处理
'''
# 数据增强
train_image_generator = ImageDataGenerator( rescale=1./200,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./200)
test_image_generator = ImageDataGenerator(rescale=1./200)


'''
生成数据
'''
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           color_mode="grayscale",
                                                           class_mode='categorical')

total_train = train_data_gen.n

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              color_mode="grayscale",
                                                              class_mode='categorical')

total_val = val_data_gen.n

test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(im_height, im_width),
                                                         color_mode="grayscale",
                                                         class_mode='categorical')

total_test = test_data_gen.n


'''
搭建CNN网络模型
'''
# 初始化模型
model = tf.keras.Sequential()

# 卷积1
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',
          input_shape=(200,200,1),activation='relu'))
# 卷积2
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',
                 activation='relu'))
# 池化
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',
          activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(2,2),padding='same',
          activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(2,2),padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(2,2),padding='same',
          activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(2,2),padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 全连接
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))

# 显示模型信息
model.summary()

'''
编译模型
'''
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

'''
开始训练
'''
reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.1,
                                patience=2,
                                mode='auto',
                                verbose=1
                             )


checkpoint = ModelCheckpoint(
                                filepath='./save_weights/pneumonia.ckpt',
                                monitor='val_acc',
                                save_weights_only=False,
                                save_best_only=True,
                                mode='auto',
                                period=1
                            )

history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=[checkpoint, reduce_lr])

'''
保存模型
'''
model.save_weights('./save_weights/pneumonia.ckpt',save_format='tf')

history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

#损失值
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# 准确率
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

'''
模型评估
'''
scores = model.evaluate(test_data_gen, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))  # 计算准确率
    misclass = 1 - accuracy  # 计算错误率
    if cmap is None:
        cmap = plt.get_cmap('Blues')  # 颜色设置成蓝色
    plt.figure(figsize=(10, 8))  # 设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示图片
    plt.title(title)  # 显示标题
    plt.colorbar()  # 绘制颜色条

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)  # x坐标标签旋转45度
        plt.yticks(tick_marks, target_names)  # y坐标

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm, 2)  # 对数字保留两位小数

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):  # 将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize:  # 标准化
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),  # 保留两位小数
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色
        else:  # 非标准化
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色

    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label')  # y方向上的标签
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))  # x方向上的标签
    plt.show()  # 显示图片


labels = ['NORMAL', 'PNEUMONIA']

# 预测验证集数据整体准确率
Y_pred = model.predict_generator(test_data_gen, total_test // batch_size + 1)
# 将预测的结果转化为one hit向量
Y_pred_classes = np.argmax(Y_pred, axis=1)
# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true=test_data_gen.classes, y_pred=Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)


