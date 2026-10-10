# 影像组学特征提取、筛选与建模设计

## 目标

基于 `Preprocess/train` 和 `Preprocess/validation` 中实际存在的患者文件夹，提取 NIfTI 影像及靶区勾画的 PyRadiomics 特征，形成统一的 `Radiomics_features.csv`；随后仅使用训练集完成特征筛选和模型调参，并在验证集进行最终评估。

## 数据关联与标签

- 患者编号以影像患者文件夹名称为准，不依赖临床表行号。
- 使用患者编号回查 `Clinical_data_cleaning.xlsx` 的 `编号`。
- 结局编码固定为：`Evaluation=1` 为 CR，`y=1`；`Evaluation=2` 为 non-CR，`y=0`。
- 不在对应影像文件夹中的临床表患者不进入影像组学建模。
- 患者级输出保留原始编号。

## ROI 选择规则

每位患者从 `structures_labels.json` 中按以下顺序选择一个实际 ROI，且排除包含 `PLAN` 的计划结构：

1. 精确名称 `GTVnx`；
2. GTVnx 命名变体，如 `GTV-nx`、`GTVNx`、`GTVnx_1`、`GTVnx-w`；
3. 其他 GTV，优先级为 `GTV` → `GTVnd` → `GTVp` → 其他 GTV；
4. `PGTVnx`；
5. 若仍无可用 ROI，记录为缺失并跳过特征提取。

选定的 ROI 原始名称、标签值、选择原因和异常信息写入特征表及异常报告。存在标签但掩膜为空、图像与掩膜无法对齐或特征提取失败时，记录异常而不静默继续。

## 特征提取

`Radiomics_feature_extraction.py` 遍历两个数据集目录，读取 `im.nii.gz`、`structures.nii.gz` 和标签 JSON。使用 3D PyRadiomics，启用形状、一阶、GLCM、GLRLM、GLSZM、GLDM、NGTDM，并启用 Wavelet 与 LoG 派生图像，以获得一千多个特征。影像和掩膜使用原始预处理后空间；不从影像重复计算临床表中已有的 GTV 体积。

## 特征筛选

`Radiomics_feature_screening.py` 只在训练集执行拟合：

1. 对训练特征拟合 Z-score 参数；
2. 按训练集 Pearson 相关系数绝对值大于 0.9 去除冗余；
3. 使用训练集标签进行 mRMR，固定选择 Top 15；
4. 使用五折交叉验证的 LASSO Logistic 回归选择正则化参数 λ；
5. 保存最终特征、标准化参数、筛选日志、LASSO 系数和 Rad-score 计算参数。

验证集只应用训练集已拟合的参数和特征列表，不参与任何筛选。

## 模型比较

`Radiomics_Machin_Learning_Modeling.py` 使用最终筛选特征比较：LR、SVM、Random Forest、LightGBM、XGBoost、ExtraTrees。模型超参数只在训练集内部五折交叉验证调节；验证集只做一次最终评估。输出模型比较、最佳模型、评价指标、阈值、校准结果、ROC/校准图、患者级预测和 Rad-score。

## 输出位置

- `outputs/final/01_临床筛选/`：特征筛选日志、最终特征和参数；
- `outputs/final/02_模型评估/`：模型比较、最终指标、图形和模型文件；
- `outputs/final/03_患者级结果/`：患者级预测、Rad-score 和异常记录；
- `outputs/archive/`：不再使用的旧结果。

根目录 `Radiomics_features.csv` 保存统一的全部影像组学特征表；相关结果目录增加 README，说明结局编码、变量、样本规模、评价指标和阈值。

## 依赖与验证

需要 `SimpleITK`、`pyradiomics`、`numpy`、`scipy`、`scikit-learn`、`lightgbm`、`xgboost`、`openpyxl`。脚本在运行前检查依赖并给出明确错误。测试覆盖 ROI 优先级、患者编号回查、训练集参数不泄漏到验证集、特征筛选输出和模型输出字段。
