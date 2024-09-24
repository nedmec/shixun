import sys
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import *
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

def dice_coefficient(y_true, y_pred, smooth=0.0000000001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def dice_coff(label, predict):
    return np.sum(2*label*predict)/(np.sum(label)+np.sum(predict))


def image_process_enhanced(img):
    img = cv2.equalizeHist(img)  # 像素直方图均衡
    return img


# 标签编码
# 标签像素点分为4类，取值分别为255，170，85，0，除以255后进行分类，分别转化为三通道
# 255->[1,0,0];170->[0,1,0];85->[0,0,1];0->[0,0,0]
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


def load_image(root, data_type, size=None, need_name_list=False, need_enhanced=False):
    image_path = os.path.join(root, data_type, "image")
    label_path = os.path.join(root, data_type, "label")
    print(image_path)

    image_list = []
    label_list = []
    image_name_list = []

    k = 0  # 如果加载全部数据在平台中训练时间过长，或出现环境崩溃等问题，建议仅加载少量数据进行训练，可将此行代码注释去掉

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
        # 对图像进行增强
        if need_enhanced is True:
            img = image_process_enhanced(img)

        img = img / 255
        label = label / 255
        image_list.append(img)
        label = label_to_code(label)  # 对标签进行编码
        label_list.append(label)

        # 如果加载全部数据在平台中训练时间过长，或出现环境崩溃等问题，建议仅加载少量数据进行训练，可将以下几行代码注释去掉
        k += 1
        if k>39:
            break

    if need_name_list is True:
        return np.array(image_list), np.array(label_list), image_name_list
    else:
        return np.array(image_list), np.array(label_list)


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

def VGG19_unet_model(input_size=(224, 224, 3), use_batchnorm=False, if_transfer=True, if_local=False):
    axis = 3
    kernel_initializer = 'he_normal'
    origin_filters = 32
    weights = None
    model_path = os.path.join(sys.path[0], 'models', 'level3_model.h5')
    print(model_path)
    if if_transfer is True:
        if if_local is True:
            weights = model_path
        else:
            weights = 'imagenet'
    vgg19 = VGG19(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg19.layers:
        layer.trainable = False
        # layer.trainable = True
    output = vgg19.layers[17].output
    print(output.shape)
    up6 = layers.Conv2D(origin_filters*8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(output))
    merge6 = layers.concatenate([vgg19.layers[13].output, up6], axis=axis)
    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)
    if use_batchnorm is True:
        conv6 = layers.BatchNormalization()(conv6)
    up7 = layers.Conv2D(origin_filters*4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([vgg19.layers[9].output, up7], axis=axis)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    if use_batchnorm is True:
        conv7 = layers.BatchNormalization()(conv7)
    up8 = layers.Conv2D(origin_filters*2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([vgg19.layers[5].output, up8], axis=axis)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    if use_batchnorm is True:
        conv8 = layers.BatchNormalization()(conv8)
    up9 = layers.Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([vgg19.layers[2].output, up9], axis=axis)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    if use_batchnorm is True:
        conv9 = layers.BatchNormalization()(conv9)
    conv10 = layers.Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(inputs=vgg19.input, outputs=conv10)
    # print(model.summary(line_length=150))
    return model

def VGG16_unet_model(input_size=(224, 224, 3), use_batchnorm=False, if_transfer=False, if_local=True):
    axis = 3
    kernel_initializer = 'he_normal'
    origin_filters = 32
    weights = None
    model_path = os.path.join(sys.path[0], 'models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print(model_path)
    if if_transfer is True:
        if if_local is True:
            weights = model_path
        else:
            weights = 'imagenet'
    vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg16.layers:
        layer.trainable = False
        # layer.trainable = True
    output = vgg16.layers[17].output
    print(output.shape)
    up6 = layers.Conv2D(origin_filters*8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(output))
    merge6 = layers.concatenate([vgg16.layers[13].output, up6], axis=axis)
    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = layers.Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)
    if use_batchnorm is True:
        conv6 = layers.BatchNormalization()(conv6)
    up7 = layers.Conv2D(origin_filters*4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([vgg16.layers[9].output, up7], axis=axis)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = layers.Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    if use_batchnorm is True:
        conv7 = layers.BatchNormalization()(conv7)
    up8 = layers.Conv2D(origin_filters*2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([vgg16.layers[5].output, up8], axis=axis)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = layers.Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    if use_batchnorm is True:
        conv8 = layers.BatchNormalization()(conv8)
    up9 = layers.Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([vgg16.layers[2].output, up9], axis=axis)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = layers.Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    if use_batchnorm is True:
        conv9 = layers.BatchNormalization()(conv9)
    conv10 = layers.Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(inputs=vgg16.input, outputs=conv10)
    # print(model.summary(line_length=150))
    return model

model = VGG16_unet_model(input_size=(224, 224, 3))
print(model.summary())

# 参数设置，命令行参数解析
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset", required=False, help='path to dataset')
    parser.add_argument('--img_enhanced', default=False, help='image enhancement')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--image-size', default=(224, 224, 3), help='the (height, width, channel) of the input image to network')
    parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--model-save', default='./models/level3_model.h5', help='folder to output model checkpoints')
    parser.add_argument('--model-path', default='./models/level3_model.h5', help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-level3", required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    return args


# 模型训练
def train_level3():
    args = get_parser()  # 获取参数
    train, train_label = load_image(args.data_root, "train", need_enhanced=args.img_enhanced)  # dataset为实际使用数据
    val, val_label = load_image(args.data_root, 'val', need_enhanced=args.img_enhanced)
    model = VGG19_unet_model(input_size=args.image_size, if_transfer=True, if_local=False)
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    model_checkpoint = callbacks.ModelCheckpoint(args.model_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train, train_label, batch_size=args.batch_size, epochs=args.niter, callbacks=[model_checkpoint],
                        validation_data=(val, val_label))
    plot_history(history, args.outf)


# 计算像素精度PA
def pixel_accuracy(label, predict):
    start_time = time.time()
    length, row, column, channels = label.shape
    true_pixel = 0
    all_pixels = length*row*column
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                if predict_cate == label_cate:
                        true_pixel = true_pixel + 1

    end_time = time.time()
    print("the pixel_accuracy is: " + str(true_pixel/all_pixels))
    print("compute pixel_accuracy use time: "+str(end_time-start_time))
    return true_pixel/all_pixels


# 计算均相似精度MPA
def mean_pixel_accuracy(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == n:
                            insaction_list[n] = insaction_list[n] + 1
                        break
    end_time = time.time()
    mean_pixel_accuracy = 0
    for i in range(class_num):
        mean_pixel_accuracy += insaction_list[i]/class_list[i]
    mean_pixel_accuracy = mean_pixel_accuracy/class_num
    print("the mean pixel accuracy is: " + str(mean_pixel_accuracy))
    print("compute mean pixel accuracy use time: "+str(end_time-start_time))
    return mean_pixel_accuracy


# 计算均交并比mIoU
def compute_mIoU(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n | predict_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == label_cate:
                            insaction_list[n] = insaction_list[n] + 1
    mIoU = 0
    for i in range(class_num):
        mIoU += insaction_list[i] / class_list[i]
    mIoU = mIoU / class_num
    end_time = time.time()
    print("the mIoU is: " + str(mIoU))
    print("compute_mIoU use time: "+str(end_time-start_time))
    return mIoU


def category(img):
    if img[0] >= 0.5:
        return 1
    elif img[1] >= 0.5:
        return 2
    elif img[2] >= 0.5:
        return 3
    else:
        return 0


# 模型测试
def predict_level3():
    args = get_parser()  # 获取参数
    test_img, test_label, test_name_list = load_image(args.data_root, "test", need_name_list=True,
                                                      need_enhanced=args.img_enhanced)
    model = load_model(args.model_path, custom_objects={'dice_coefficient': dice_coefficient,
                                                        'dice_coefficient_loss': dice_coefficient_loss})
    result = model.predict(test_img)

    dc = dice_coff(test_label, result)
    print("the dice coefficient is: " + str(dc))
    pixel_accuracy(test_label, result)
    mean_pixel_accuracy(test_label, result)
    compute_mIoU(test_label, result)
    for i in range(result.shape[0]):
        final_img = tensorToimg(result[i])
        ori_img = test_img[i]
        ori_gt = tensorToimg(test_label[i])

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
    train_level3()
    # predict_level3()
    print("time:", time.time()-s_t)