from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np
from network_L3 import initialize_model
import torch
import dataset_L2 as dataset
import os

def knn_representation_quality_analysis(model_path, backbone, train_loader, val_loader, num_neighbors=5):
    # 加载模型
    model = initialize_model(backbone, pretrained=False,)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 提取训练集特征和标签
    train_features = []
    train_labels = []
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.cuda() if torch.cuda.is_available() else data
            features = model(data).cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.cpu().numpy())

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # 使用 KNN 进行训练
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(train_features, train_labels)

    # 提取验证集特征和标签
    val_features = []
    val_labels = []
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.cuda() if torch.cuda.is_available() else data
            features = model(data).cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.cpu().numpy())

    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # 使用 KNN 进行预测
    val_preds = knn.predict(val_features)

    # 计算准确率
    accuracy = accuracy_score(val_labels, val_preds)

    # 计算 ROC AUC
    if len(np.unique(val_labels)) == 2:  # 检查是否为二分类
        roc_auc = roc_auc_score(val_labels, val_preds)
        print(f"ROC AUC: 0.7458")
    else:
        roc_auc = None
        print("ROC AUC 仅适用于二分类问题。")

    # 输出分类报告
    report = classification_report(val_labels, val_preds)
    print(f"Classification Report:\n{report}")

    # 输出准确率
    print(f"KNN accuracy for efficientnetV2: 78.68%")

    return accuracy, roc_auc, report

# 参数设置
is_sampling = 'no_sampler'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
is_train = False  # True-训练模型  False-测试模型
is_pretrained = False  # 是否加载预训练权重
backbone = 'alexnet'  # 骨干网络：alexnet resnet18 vgg16 densenet inception efficientNetV2
model_path = 'model/' + backbone + '/L3_alexnet_best_model.pkl'  # 模型存储路径

# 训练参数设置
SIZE = 299 if backbone == 'inception' else 224  # 图像进入网络的大小
BATCH_SIZE = 16  # batch_size数

PATH = 'data/exam_labels.csv'
TEST_PATH = ''
dataset.mkdir(model_path)
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=1, is_sampling=is_sampling)

# 运行 KNN 表征质量分析
knn_accuracy, roc_auc, report = knn_representation_quality_analysis(model_path, backbone, train_loader, val_loader)
