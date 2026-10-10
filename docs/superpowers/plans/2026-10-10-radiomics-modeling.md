# 影像组学建模实施计划

## 目标

创建三个可重复运行的脚本，完成影像组学特征提取、训练集特征筛选、模型比较与验证集最终评估，并输出可追溯的患者级结果和说明文档。

## 文件清单

- 新增 `Code/Radiomics_feature_extraction.py`：遍历患者文件夹、选择 ROI、提取 PyRadiomics 特征。
- 新增 `Code/Radiomics_feature_screening.py`：按训练集拟合标准化、Pearson、mRMR Top 15 和 LASSO。
- 新增 `Code/Radiomics_Machin_Learning_Modeling.py`：训练集内部调参、验证集最终评估、Rad-score 与患者级预测。
- 新增 `Code/test_radiomics_pipeline.py`：测试 ROI 选择、编号回查和筛选/建模输出接口。
- 更新根目录 `README.md`：补充影像组学运行顺序、结局编码和结果说明。
- 新增/更新 `outputs/final/01_临床筛选/README.md`、`02_模型评估/README.md`、`03_患者级结果/README.md`。

## 执行步骤

### 1. 依赖与数据接口

- 检查并安装/提示 `scipy`、`scikit-learn`；保留已有 PyRadiomics、LightGBM、XGBoost 等依赖。
- 用文件夹名称生成训练/验证患者集合。
- 从临床表按 `编号` 建立映射，过滤无有效 `Evaluation` 的患者。

### 2. 特征提取

- 实现 ROI 优先级：精确 `GTVnx`、GTVnx 变体、`GTV`、`GTVnd`、`GTVp`、其他 GTV、`PGTVnx`。
- 排除 `PLAN` 结构，检查标签存在性、掩膜非空性和空间一致性。
- 配置 3D PyRadiomics，启用原始图像、Wavelet 和 LoG，并启用形状/一阶/纹理类别。
- 逐患者输出阶段进度、耗时、预计剩余时间；失败患者进入异常 CSV/JSON。
- 写出根目录 `Radiomics_features.csv`，包含编号、split、ROI 元数据和特征列。

### 3. 特征筛选

- 仅读取实际存在于训练/验证目录且能回查临床结局的患者。
- 在训练集拟合 Z-score，删除零方差特征。
- 按绝对 Pearson 相关系数 >0.9 逐步删除冗余特征。
- 用训练集互信息/相关性实现 mRMR，固定取 Top 15。
- 用五折交叉验证选择 LASSO Logistic 的最佳 λ，并保存标准化参数、特征列表、系数和筛选日志。
- 生成训练集和验证集 Rad-score 表，但不使用验证集参与筛选。

### 4. 模型比较

- 使用最终筛选特征构建 LR、SVM、Random Forest、LightGBM、XGBoost、ExtraTrees。
- 训练集内部使用五折交叉验证调参；不根据验证集调参。
- 在验证集输出 ROC-AUC、PR-AUC、灵敏度、特异度、准确率、Brier score、校准指标。
- 保存最佳模型、比较表、阈值、ROC/校准图和患者级预测。

### 5. 验证标准

- 单元测试覆盖：ROI 优先级、PLAN 排除、患者编号回查、训练参数不泄漏、缺失患者记录。
- 用小型合成 NIfTI/掩膜测试特征提取接口；真实数据运行完成后检查行数、患者编号集合和特征列数。
- 运行三个脚本的完整流程，确认输出文件存在、CSV 可读、训练/验证样本数和异常数有记录。
- 更新 README，记录结局编码、使用变量、数据集规模、评价指标和阈值定义。
