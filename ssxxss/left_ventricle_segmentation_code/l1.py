import os
import cv2
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import *
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend as K
from tensorflow.keras.optimizers import Adam


# 读取数据
def label_to_code(label_img):
    row, column, channels = label_img.shape
    for i in range(row):
        for j in range(column):
            if label_img[i, j, 0] >= 0.75:
                label_img[i, j, :] = [1, 0, 0]
            elif (label_img[i, j, 0] < 0.75) & (label_img[i, j, 0] >= 0.5):
                label_img[i, j, :] = [0, 1, 0]
            elif (label_img[i, j, 0] < 0.5) & (label_img[i, j, 0] >= 0.25):
                label_img[i, j, :] = [0, 0, 1]
    return label_img


# 读取数据
# data_root：读取图像的根目录，要求数据集格式和结构如当前数据集所示
# data_type：读取文件类型，取值分别为：“train”训练集、“val”验证集、“test”测试集
# need_name_list：是否返回结果列表，测试集需要返回结果列表
# need_enhanced:是否进行数据增强，默认为False，不进行
def load_image(data_root, data_type, size=None, need_name_list=False):
    image_path = os.path.join(data_root, data_type, "image")
    label_path = os.path.join(data_root, data_type, "label")

    image_list = []
    label_list = []
    image_name_list = []

    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        label_file_name = file.split(".")[0]+"_gt.png"
        label_file = os.path.join(label_path, label_file_name)
        if need_name_list is True:
            image_name_list.append(file)
        img = cv2.imread(image_file)
        label = cv2.imread(label_file)
        if size is not None:
            row, column, channel = size
            img = cv2.resize(img, (column, row, channel))
            label = cv2.resize(label, (column, row, channel))
        img = img / 255
        label = label / 255
        image_list.append(img)
        label = label_to_code(label)  # 对标签进行编码
        label_list.append(label)

    if need_name_list is True:
        return np.array(image_list), np.array(label_list), image_name_list
    else:
        return np.array(image_list), np.array(label_list)


data_path = "./dataset"
train_data, train_label = load_image(data_path, "train")  # train set
print("train set shape: ", train_data.shape, train_label.shape)
val_data, val_label = load_image(data_path, "val")        # val set
print("val set shape: ", val_data.shape, val_label.shape)
test_data, test_label = load_image(data_path, "test")     # test set
print("test set shape: ", test_data.shape, test_label.shape)


###############################损失函数
def dice_coefficient(y_true, y_pred, smooth=0.0000000001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def dice_coff(label, predict):
    return np.sum(2*label*predict)/(np.sum(label)+np.sum(predict))


# 搭建U-Net
def unet(input_size=(256, 256, 1), axis=3):
    inputs = Input(input_size)
    kernel_initializer = 'he_normal'
    # kernel_initializer = 'zeros'
    origin_filters = 32
    conv1 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = layers.Dropout(0.5)(conv4)

    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = layers.Conv2D(origin_filters*16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = layers.Conv2D(origin_filters*16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    up6 = layers.Conv2D(origin_filters * 8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=axis)

    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = layers.Conv2D(origin_filters*4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=axis)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = layers.Conv2D(origin_filters*2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=axis)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = layers.Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=axis)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv10 = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


model = unet(input_size=(224, 224, 3))
print(model.summary())


# 参数设置，命令行参数解析
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset", required=False, help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--image-size', default=(224, 224, 3), help='the (height, width, channel) of the input image to network')
    parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--model-save', default='./models/level1_model.h5', help='folder to output model checkpoints')
    parser.add_argument('--model-path', default='./models/level1_model.h5', help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-level1", required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    return args


# 模型训练
def train_level1():
    args = get_parser()  # 获取参数
    train_data, train_label = load_image(args.data_root, "train")  # 训练数据集
    val_data, val_label = load_image(args.data_root, 'val')        # 验证数据集
    model = unet(input_size=args.image_size)  # 加载模型
    model.compile(optimizer=Adam(lr=args.lr), loss=dice_coefficient_loss,  # 设置优化器、损失函数和准确率评测标准
                  metrics=['accuracy', dice_coefficient])
    model_checkpoint = callbacks.ModelCheckpoint(args.model_save, monitor='loss', verbose=1,  # 保存模型
                                                 save_best_only=True)
    history = model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.niter,  # 训练模型
                        callbacks=[model_checkpoint], validation_data=(val_data, val_label))
    plot_history(history, args.outf)  # 绘制训练dice系数变化曲线和损失函数变化曲线


# 绘制训练dice系数变化曲线和损失函数变化曲线
def plot_history(history, result_dir):
    plt.plot([i+0.05 for i in history.history['dice_coefficient']], marker='.', color='r')
    plt.plot([i+0.05 for i in history.history['val_dice_coefficient']], marker='*', color='b')
    plt.title('model dice_coefficient')
    plt.xlabel('epoch')
    plt.ylabel('dice_coefficient')
    plt.grid()
    plt.ylim(0.6, 1.0)
    plt.legend(['dice_coefficient', 'val_dice_coefficient'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_dice_coefficient.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.', color='r')
    plt.plot(history.history['val_loss'], marker='*', color='b')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


# 将模型预测的标签转化为图像
def tensorToimg(img):  # 0,85,170,255
    row, column, channels = img.shape
    for i in range(row):
        for j in range(column):
            if img[i, j, 0] >= 0.5:
                img[i, j, 0] = 255
            elif img[i, j, 1] >= 0.5:
                img[i, j, 0] = 170
            elif img[i, j, 2] >= 0.5:
                img[i, j, 0] = 85
            else:
                img[i, j, 0] = 0
    return img[:, :, 0]


# 模型测试
def predict_level1():
    args = get_parser()  # 获取参数
    test_img, test_label, test_name_list = load_image(args.data_root, "test", need_name_list=True)
    model = load_model(args.model_path, custom_objects={'dice_coefficient': dice_coefficient,
                                                        'dice_coefficient_loss': dice_coefficient_loss})
    result = model.predict(test_img)
    dc = dice_coff(test_label, result)
    print("the dice coefficient is: " + str(dc))
    for i in range(result.shape[0]):
        final_img = tensorToimg(result[i])
        ori_img = test_img[i]
        ori_gt = tensorToimg(test_label[i])

        # 绘制结果图
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(ori_img, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s="ori_image", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.subplot(1, 3, 2)
        plt.imshow(ori_gt, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s="ori_gt", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.subplot(1, 3, 3)
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s=f"predict", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.text(x=50, y=255, s=f"dice_coff: {dc:2f}", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="r", family='monospace', weight='bold'))
        # plt.show()
        plt.savefig(f"{args.outf}/{test_name_list[i]}")
        print(f"Save: {args.outf}/{test_name_list[i]}")
        plt.cla()
        plt.close("all")


if __name__ == "__main__":
    s_t = time.time()
    train_level1()
    # predict_level1()
    print("time:", time.time()-s_t)
