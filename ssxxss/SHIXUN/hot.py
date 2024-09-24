import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import cv2


def calculate_pixel_intensity(img):
    """
    计算图像的像素强度分布。

    输入:
        img: 输入图像数组。

    返回:
        pixel_intensity: 图像的像素强度分布。
    """
    # 将图像展平为一维数组
    pixel_values = img.flatten()

    # 计算像素强度的分布
    pixel_intensity, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))

    return pixel_intensity


def plot_heatmap(pixel_intensity):
    """
    绘制像素强度的热力图。

    输入:
        pixel_intensity: 图像的像素强度分布。
    """
    # 将像素强度数据重塑为2D数组（可以自定义行列数）
    pixel_intensity_2d = np.reshape(pixel_intensity, (16, 16))  # 这里16x16只是一个示例，实际可根据需要调整

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pixel_intensity_2d, cmap='viridis', annot=True, fmt='d')
    plt.title('Pixel Intensity Heatmap')
    plt.xlabel('Intensity Bin')
    plt.ylabel('Intensity Bin')
    plt.show()


def process_dicom_image(dicom_path):
    """
    处理DICOM图像并绘制像素强度热力图。

    输入:
        dicom_path: DICOM图像文件路径。
    """
    # 读取DICOM图像
    dcm = pydicom.read_file(dicom_path)
    img = dcm.pixel_array

    # 计算像素强度
    pixel_intensity = calculate_pixel_intensity(img)

    # 绘制热力图
    plot_heatmap(pixel_intensity)


# 示例使用
dicom_path = 'data/images/00001.dcm'
process_dicom_image(dicom_path)