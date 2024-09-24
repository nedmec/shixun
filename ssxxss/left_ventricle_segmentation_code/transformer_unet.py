from keras.models import *
from keras.layers import *
from keras.optimizers import *
from model_code.loss_function import dice_coefficient_loss, dice_coefficient
from keras.applications.vgg16 import VGG16
import os
import sys

def VGG16_unet_model(input_size=(224, 224, 3), use_batchnorm=False, if_transfer=False, if_local=True):
    axis = 3
    kernel_initializer = 'he_normal'
    origin_filters = 32
    weights = None
    model_path = os.path.join(sys.path[0], 'model_code', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print(model_path)
    if if_transfer is True:
        if if_local is True:
            weights = model_path
        else:
            weights = 'imagenet'
    vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg16.layers:
        layer.trainable = True
    output = vgg16.layers[17].output
    up6 = Conv2D(origin_filters * 8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(output))
    merge6 = concatenate([vgg16.layers[13].output, up6], axis=axis)
    conv6 = Conv2D(origin_filters * 8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(origin_filters * 8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)
    if use_batchnorm is True:
        conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(origin_filters * 4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([vgg16.layers[9].output, up7], axis=axis)
    conv7 = Conv2D(origin_filters * 4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(origin_filters * 4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    if use_batchnorm is True:
        conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(origin_filters * 2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([vgg16.layers[5].output, up8], axis=axis)
    conv8 = Conv2D(origin_filters * 2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(origin_filters * 2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    if use_batchnorm is True:
        conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([vgg16.layers[2].output, up9], axis=axis)
    conv9 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    if use_batchnorm is True:
        conv9 = BatchNormalization()(conv9)
    
    # 降低空间维度
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    pool = MaxPooling2D(pool_size=(4, 4))(conv10)  # 将尺寸从 (224, 224) 降到 (56, 56)
    
    # 调整形状以适配 MultiHeadAttention
    reshape = Reshape((pool.shape[1]*pool.shape[2], pool.shape[3]))(pool)  # (batch_size, seq_len, channels)
    
    # 添加 MultiHeadAttention 层
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(reshape, reshape)
    add = Add()([reshape, attention_output])  # 残差连接
    
    # 恢复形状
    reshape_back = Reshape((pool.shape[1], pool.shape[2], pool.shape[3]))(add)
    
    # 上采样回原始尺寸
    upsample = UpSampling2D(size=(4, 4))(reshape_back)
    
    # 最后的卷积层
    conv11 = Conv2D(32, 3, activation='relu', padding='same')(upsample)
    conv12 = Conv2D(3, 1, activation='sigmoid', padding='same')(conv11)
    
    model = Model(inputs=vgg16.input, outputs=conv12)
    print(model.summary(line_length=150))
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coefficient_loss, metrics=[dice_coefficient])

    return model

if __name__ == '__main__':
    VGG16_unet_model(use_batchnorm=True)
