# Perceiver-IO Zero-Shot Anomaly Detection (MVTec-AD)

本项目基于 [`krasserm/perceiver-io`](https://github.com/krasserm/perceiver-io) 扩展，实现 **Zero-Shot Anomaly Detection**（零样本异常检测），目标覆盖：

- **Pixel-level anomaly localization + segmentation**
- **Image-level anomaly scoring**
- 后续支持 **LoRA 参数高效微调**

---

## 1. 项目目标

### 最终目标
构建一个可训练、可评估、可复现实验的异常检测系统，满足：

1. 使用 MVTec-AD（15 类工业缺陷数据）
2. 训练仅使用正常样本（normal-only）
3. 测试在未见类别上进行异常检测（Zero-Shot）
4. 输出：
   - 像素级异常图 `(B, H, W, 1)`
   - 图像级异常分数 `(B,)`
5. 支持后续 LoRA 微调以降低训练参数量并提升泛化效率

---

## 2. 当前阶段完成情况（已完成）

## 2.1 数据层（MVTec DataModule）
已新增：

- `perceiver/data/vision/mvtec.py`
- `perceiver/data/vision/__init__.py`（导出 `MVTecDataModule`）

已实现能力：

- MVTec 类别枚举（15 类）
- `train_categories / test_categories` 分离（Zero-Shot 划分基础）
- 训练集仅采样 `train/good`
- 验证/测试读取 `test/good + test/<defect_type>`
- 异常样本自动尝试匹配 `ground_truth/<defect_type>/<stem>_mask.*`
- 输出字段统一：
  - `image`
  - `label`（0/1）
  - `mask`（像素级）
  - `category`
  - `defect_type`
  - `path`
- 支持：
  - resize
  - ImageNet normalize
  - channels-last
  - 基础数据增强（训练时）

---

## 2.2 模型层（Perceiver 扩展）
已新增：

- `perceiver/model/vision/anomaly_detector/backend.py`
- `perceiver/model/vision/anomaly_detector/__init__.py`

已实现能力：

- 复用 `PerceiverEncoder / PerceiverDecoder`
- `AnomalyImageInputAdapter`
  - 输入 `(B,H,W,C)` -> `(B,H*W,C')`
  - 叠加 Fourier 位置编码
- `SpatialQueryProvider`
  - 生成空间查询（默认 `map_shape=(64,64)`）
- `AnomalyMapOutputAdapter`
  - 输出低分辨率异常图
  - 双线性上采样到原始图像尺寸
  - 输出 `anomaly_logits`, `anomaly_prob`, `image_score`

---

## 2.3 训练层（Lightning）
已新增：

- `perceiver/model/vision/anomaly_detector/lightning.py`

已实现能力：

- `LitAnomalyDetector` 训练闭环完整
- 核心 loss 对接：
  - `outputs["anomaly_logits"]` vs `batch["mask"]`（pixel-level BCEWithLogits）
- 可选 image-level loss（默认权重 0）
- 兼容 mask shape：
  - `(B,H,W,1)` 或 `(B,1,H,W)`
- 指标：
  - pixel AUROC
  - image AUROC（可选）
- 已完成 1 epoch 可执行训练验证（从日志确认）

---

## 2.4 脚本入口层（CLI）
已新增：

- `perceiver/scripts/vision/anomaly_detector.py`

已实现能力：

- 可通过 `python -m perceiver.scripts.vision.anomaly_detector fit ...` 启动
- 已 link：
  - `data.image_shape -> model.encoder.image_shape`
- 已配置一组 anomaly detector 默认参数（latents、decoder map、loss 权重等）

---

## 3. 当前运行状态（阶段评估）

## 已验证通过
- 模型可实例化
- 数据模块可被 CLI 识别
- 训练可跑完整 epoch
- 验证阶段能产出 `val_loss` 与 `val_pixel_auroc`

## 观察到的现象
- 目前 AUROC 偏低（例如 ~0.2），属于 baseline 初期现象
- 当前主要是 pipeline 已打通，性能仍需系统优化

---

## 4. 关键代码说明

## `perceiver/data/vision/mvtec.py`
核心职责：

- 扫描目录
- 构建样本索引
- 按 split 返回 Dataset
- DataLoader 封装

关键函数：

- `_collect_train_samples_for_category()`: 仅 `train/good`
- `_collect_test_samples_for_category()`: `test/good + test/defect`
- `_find_mask_path()`: ground-truth mask 映射
- `setup()`: 构建 `ds_train / ds_val / ds_test`

---

## `perceiver/model/vision/anomaly_detector/backend.py`
核心职责：

- 定义 anomaly detector backend
- 输入适配 + 空间 query + 输出 head

关键类：

- `AnomalyEncoderConfig`
- `AnomalyDecoderConfig`
- `AnomalyImageInputAdapter`
- `SpatialQueryProvider`
- `AnomalyMapOutputAdapter`
- `AnomalyDetector`

输出标准：

- `anomaly_logits_lowres`
- `anomaly_logits`
- `anomaly_prob`
- `image_score`

---

## `perceiver/model/vision/anomaly_detector/lightning.py`
核心职责：

- 定义训练 step、日志、验证指标
- 连接 DataModule 字段与模型输出字段

关键点：

- `step()` 中 shape 校验严格
- pixel-level loss 是当前主要监督信号
- 指标 epoch-end 汇总并 reset

---

## `perceiver/scripts/vision/anomaly_detector.py`
核心职责：

- Lightning CLI 入口
- 参数默认值
- data-model 参数联动

---



---

## 5. 下一阶段计划（从现在到最终目标）

## 阶段 A：基线稳态训练（短期）
目标：得到可信 baseline。

- 固定 zero-shot split（显式指定 train/test categories）
- 训练 10~30 epochs（非 `fast_dev_run`）
- 记录：
  - `val_pixel_auroc`
  - `val_image_auroc`
  - loss 曲线
- 加入简单可视化（保存 anomaly map）

## 阶段 B：指标与评估体系完善
目标：建立标准评估报告。

- 增加测试阶段指标脚本
- 分类输出：
  - image-level AUROC
  - pixel-level AUROC
- 可选补充：
  - AUPRO / F1@threshold
- 分类别报告（15 类平均 + 每类）

## 阶段 C：LoRA 微调接入
目标：参数高效训练。

- 新增 `perceiver/model/core/lora.py`
- LoRA 注入位置：
  - attention `q_proj`, `v_proj`（优先）
- 冻结 backbone，仅训练：
  - LoRA 参数
  - anomaly head
- 做 rank 对比（r=4, r=8）

## 阶段 D：工程化与复现
目标：稳定可交付。

- 完善 `examples/anomaly/train.py` + `train.sh`
- 增加配置模板（YAML）
- 固化实验命令与日志目录规范
- 补充 README 结果表格与示意图

---

