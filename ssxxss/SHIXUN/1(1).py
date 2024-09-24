import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载并预处理图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# 2. 加载预训练的VGG16模型
def load_model():
    model = vgg16(pretrained=True)
    model.eval()
    return model

# 3. 提取特征
def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model.features(image_tensor)
    return features

# 4. 生成热力图
def generate_heatmap(features):
    # 取出最后一层卷积层的输出
    feature_map = features[0].cpu().numpy()  # 转为 NumPy 数组
    feature_map = np.mean(feature_map, axis=0)  # 计算每个通道的均值
    feature_map = np.max(feature_map, axis=0)  # 计算每个位置的最大值

    # 归一化到 [0, 1] 范围
    feature_map -= feature_map.min()
    feature_map /= feature_map.max()

    # 显示热力图
    plt.imshow(feature_map, cmap='hot')
    plt.colorbar()
    plt.show()

# 主函数
def main(image_path):
    image_tensor = load_image(image_path)
    model = load_model()
    features = extract_features(model, image_tensor)
    generate_heatmap(features)

# 运行示例
if __name__ == '__main__':
    image_path = r'C:\Users\86193\Desktop\SHIXUN\data\images'  # 替换为你的图像路径
    main(image_path)




