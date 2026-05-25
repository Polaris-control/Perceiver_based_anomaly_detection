
# Perceiver 异常检测：面积正则化与排序优化

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**基于合成监督的 Perceiver IO 异常定位 – 面向跨类别工业缺陷检测的区域激活校准与排序优化**

本项目提供了一个完整的 **零样本/单类别异常检测** 流程，采用 **Perceiver IO** 作为主干网络，结合预训练分类器、LoRA 高效微调、合成异常生成，以及 **前景面积正则化** 和 **排序损失** 的组合策略，显著提升像素级 AUROC 与异常面积校准能力。

---

## 📌 目录

- [项目概述](#项目概述)
- [主要特点](#主要特点)
- [模型架构](#模型架构)
- [实验结果](#实验结果)
- [安装与依赖](#安装与依赖)
- [使用方法](#使用方法)
  - [1. 预训练分类器（可选）](#1-预训练分类器可选)
  - [2. 异常检测训练](#2-异常检测训练)
  - [3. 评估与可视化](#3-评估与可视化)
- [项目结构](#项目结构)
- [引用](#引用)
- [致谢](#致谢)

---

## 项目概述

工业缺陷检测中，异常样本极为稀缺且形态多样，传统监督学习方法难以应用。本项目仅使用**正常图像**进行训练，并在 **MVTec AD** 数据集上进行 **跨类别评估**（前10类训练，后5类测试），以检验模型的泛化能力。核心贡献包括：

- **预训练分类器**（10类正常图像，验证准确率 99.8%），获得高质量编码器特征。
- **LoRA 微调**（仅更新编码器 0.5% 参数），轻量适配异常检测任务。
- **五种合成异常**（CutPaste、亮度扰动、模糊、纹理噪声、裂纹）及不规则多边形 mask。
- **前景面积正则化**（Smooth L1 损失），约束预测异常面积与真实面积一致，缓解全局概率塌缩。
- **排序损失**（Ranking Loss）直接优化像素级 AUROC，增强异常与正常区域的得分分离。

在跨类别设定下，像素级 AUROC 从 **0.635 稳定提升至 0.738**，面积相关系数从 **0.00 提升至 0.45**，训练波动显著减小。

---

## 主要特点

-  **仅使用正常图像训练**，无需真实缺陷样本
-  **跨类别评估**（训练10类，测试5类），严格检验泛化能力
-  **Perceiver IO 主干** – 线性复杂度、灵活解码、保留像素级位置信息
-  **LoRA** – 参数高效微调，避免过拟合
-  **5种合成异常** + 不规则多边形 mask，增强数据多样性
-  **面积正则化 + 排序损失**，同时提升校准与 AUROC 稳定性
-  **详细日志记录**（梯度、预测分布、面积相关性等）
-  **丰富的可视化工具**（热力图、ROC 曲线、logits 直方图）

---

## 模型架构

```
输入图像 (224×224 RGB)
    ↓
傅里叶位置编码
    ↓
交叉注意力 → 潜在阵列 (512 × 1024) → 多层自注意力 (×4)
    ↓
解码器交叉注意力（空间查询，112×112 或 224×224）
    ↓
卷积细化模块（标准卷积 + 空洞卷积）
    ↓
上采样 → 异常 logits (224×224)
    ↓
Sigmoid → 概率图
    ↓
Top‑k 池化 → 图像级分数
```

### LoRA 注入
在编码器的所有多头注意力层的 `q_proj`、`k_proj`、`v_proj` 中添加低秩适配（rank=8）。编码器仅约 215k 参数可训练（占总量的 0.5%）。

---

## 实验结果

**数据集**：MVTec AD  
**训练类别**：bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw  
**测试类别**：tile, toothbrush, transistor, wood, zipper（跨类别）

| 配置 | 像素 AUROC | 面积相关性 | AUROC 标准差 |
|------|-----------|-----------|--------------|
| 随机初始化 + BCE | 0.635 | 0.02 | 0.12 |
| + 预训练编码器 | 0.670 | 0.10 | 0.10 |
| + 面积损失 + 低 Dice | 0.695 | 0.25 | 0.08 |
| + 排序损失 | 0.722 | 0.38 | 0.05 |
| **完整模型（+ 超参数调优）** | **0.738** | **0.45** | **0.04** |

- **面积相关性** 从 0.00 提升至 **0.45**，表明预测异常面积开始随真实缺陷大小单调变化。
- **AUROC 波动** 显著减小（标准差从 0.12 降至 0.04）。

### 可视化示例
- logits 分布直方图（正常 vs 异常像素）改进前后对比明显分离。
- ROC 曲线更陡峭，AUC 稳定在 0.74 左右。
- 热力图对比：改进后模型对大面积缺陷响应增强，面积校准改善。

---

## 安装与依赖

### 环境要求
- Python 3.8+
- PyTorch 1.10+
- CUDA（推荐，非必需）

### 创建环境
```bash
conda create -n perceiver python=3.8
conda activate perceiver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 安装依赖
```bash
pip install -r requirements.txt
```

克隆仓库：
```bash
git clone https://github.com/Polaris-control/Perceiver_based_anomaly_detection.git
cd Perceiver_based_anomaly_detection
```

---

## 使用方法

### 1. 预训练分类器（可选）

如果您希望获得与项目相同的预训练编码器权重，可以运行：
```bash
python examples/training/img_clf/train_my_clf.py
```
训练日志和 checkpoint 将保存在 `logs/mvtec_224_10class_pretrain_fixed/` 目录下。

### 2. 异常检测训练

修改 `examples/anomaly/train.py` 中的预训练权重路径（或使用项目提供的最佳模型），然后运行：
```bash
python examples/anomaly/train.py
```

**关键超参数**（位于 `train.py`）：
```python
pixel_pos_weight = 30.0          # 正样本权重
focal_alpha = 0.75               # Focal Loss alpha
dice_loss_weight = 0.05
area_loss_weight = 1.0
ranking_loss_weight = 0.3
decoder_lr = 1e-4
lora_lr = 5e-5
```

### 3. 评估与可视化

评估训练好的异常检测模型并生成可视化结果：
```bash
python examples/anomaly/test.py
```
该脚本会：
- 计算像素级和图像级 AUROC。
- 生成异常热力图、ROC 曲线、logits 直方图、面积相关性散点图。
- 结果保存在 `vis_results/` 目录下。

评估分类器（混淆矩阵、t‑SNE 等）：
```bash
python examples/anomaly/test_classifier.py
```

---

## 项目结构

```
perceiver-io/
├── examples/
│   ├── anomaly/
│   │   ├── train.py                # 异常检测训练
│   │   ├── test.py                 # 评估与可视化
│   │   └── test_classifier.py      # 分类器评估
│   └── training/img_clf/
│       └── train_my_clf.py         # 分类器预训练
├── perceiver/
│   ├── model/vision/
│   │   ├── image_classifier/       # 分类器模型
│   │   └── anomaly_detector/
│   │       ├── backend.py          # Perceiver IO 模型定义
│   │       ├── lightning.py        # PyTorch Lightning 模块（含损失函数）
│   │       ├── lora.py             # LoRA 注入
│   │       └── metrics.py          # AUROC 工具
│   └── data/vision/
│       ├── mvtec.py                # 原始 MVTec 数据集（异常检测）
│       └── mvtec_category.py       # 分类数据集（10 类）
├── logs/                           # 训练日志与 checkpoint
└── vis_results/                    # 可视化输出
```

---

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{perceiver_anomaly_2025,
  author       = {Polaris-control},
  title        = {Perceiver-based Anomaly Detection with Area Regularization and Ranking Loss},
  year         = {2025},
  url          = {https://github.com/Polaris-control/Perceiver_based_anomaly_detection},
  note         = {Cross-category industrial defect detection on MVTec AD}
}
```

---

