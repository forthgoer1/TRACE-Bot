# TRACE-Bot

本项目是一个基于多特征融合的社交机器人检测模型，通过整合用户行为序列、AIGC检测评分和用户个人信息等多种特征，实现高精度的社交机器人检测。

## 项目结构

```
TRACE-Bot/
├── src/              # 源代码目录
│   ├── data_process.py         # 数据处理和清洗
│   ├── behavior_sequence.py    # 行为序列提取
│   ├── GLTR_detection.py       # GLTR模型检测
│   ├── fast_detectgpt.py        # Fast DetectGPT模型检测
│   ├── feature_integration.py  # 特征整合
│   └── fusion_detection.py       # 特征融合和模型检测
├── README.md         # 项目说明
└── requirements.txt  # 依赖项
```

## 功能模块

### 1. 数据处理和清洗 (`src/data_process.py`)
- 处理NDJSON格式数据
- 转换为CSV格式
- 展开嵌套的JSON数据结构
- 数据清洗和预处理

### 2. 行为序列提取 (`src/behavior_sequence.py`)
- 提取用户的推文类型（原创、转发、回复）
- 构建行为序列
- 计算序列压缩率等特征

### 3. AIGC检测 - GLTR (`src/GLTR_detection.py`)
- 使用BERT和GPT-2模型计算文本概率
- 生成GLTR检测评分作为特征

### 4. AIGC检测 - Fast DetectGPT (`src/fast_detectgpt.py`)
- 使用Fast DetectGPT模型检测文本
- 生成检测评分作为特征

### 5. 特征整合 (`src/feature_integration.py`)
- 提取用户个人信息特征
- 整合所有特征为一个特征数据文件

### 6. 特征融合和模型检测 (`src/fusion_detection.py`)
- 使用GPT-2作为文本语义编码器
- 融合行为特征和文本特征
- 训练和评估社交机器人检测模型


## 数据集

### Fox8-23
> https://zenodo.org/records/8035290

### BotSim-24
> https://github.com/QQQQQQBY/BotSim/tree/main/BotSim-24-Dataset


## 运行流程

1. **数据处理**：运行 `src/data_process.py` 处理原始数据
2. **行为序列提取**：运行 `src/behavior_sequence.py` 提取行为序列特征
3. **AIGC检测**：分别运行 `src/GLTR_detection.py` 和 `src/fast_detectgpt.py` 生成AIGC检测特征
4. **特征整合**：运行 `src/feature_integration.py` 整合所有特征
5. **模型训练和检测**：运行 `src/feature_fusion.py` 训练模型并进行检测

## 依赖项

- Python 3.8+
- pandas
- numpy
- torch
- transformers
- scikit-learn
- tqdm

## 模型性能

在测试集上的性能指标：
- Accuracy: 0.9846
- Precision: 0.9825
- Recall: 0.9868
- F1-score: 0.9847
- ROC-AUC: 0.9986

## 注意事项

- 运行AIGC检测模型需要较大的计算资源，建议在GPU环境下运行
- 首次运行时会自动下载所需的预训练模型
- 数据处理和特征提取步骤可能需要较长时间，具体取决于数据规模
