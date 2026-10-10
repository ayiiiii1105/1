# 临床结局预测项目

## 项目目的

使用临床变量预测鼻咽癌患者达到完全缓解（CR）的概率。

- `Evaluation=1`：CR，建模标签 `y=1`
- `Evaluation=2`：non-CR，建模标签 `y=0`
- 预测概率统一表示患者达到 CR 的概率

## 运行顺序

在项目根目录打开 VSCode 终端，依次运行：

```powershell
python Code\Data_cleaning.py
python Code\Clinical_data_screening.py
python Code\Clinical_Machine_Learning_Modeling.py
```

当前建模流程为：训练集拟合全部模型，在合并验证集比较 ROC-AUC，选择最佳模型，再在同一合并验证集选择 Youden 阈值并报告结果。当前没有独立测试集。

## 目录说明

```text
Code/
├─ 正式管线、探索对照与测试脚本（见 FILES.md）
logs/
└─ 运行日志
outputs/final/
├─ 01_临床筛选/
├─ 02_模型评估/
└─ 03_患者级结果/
outputs/exploratory/
└─ 探索性对照产物
outputs/archive/
└─ 旧版或不再使用的结果
```

完整文件清单（代码 / 正式产出 / 中间产物 / 可清理项）见 `FILES.md`。

## 最重要的结果文件

- `outputs/final/02_模型评估/best_model_holdout_metrics.csv`：最终合并验证集性能。
- `outputs/final/02_模型评估/validation_model_comparison.csv`：全部模型比较。
- `outputs/archive/pre_binwidth_fix/calibration_curve.png`：校准曲线（旧版临床流程，早于最新重跑）。
- `outputs/final/03_患者级结果/clinical_patient_predictions.csv`：患者 CR 概率和临床评分。
- `outputs/final/03_患者级结果/training_leave_one_out_influence.csv`：训练集留一法影响分析。

## 数据说明

- 原始临床表：`Clinical_data.xlsx`
- 清洗后临床表：`Clinical_data_cleaning.xlsx`
- 患者原始编号与影像文件夹编号保持一致。
- GTV 体积已经补充到原始临床表，数据清洗脚本只复制该值，不重复计算。

## 结果解读提醒

合并验证集同时用于模型比较和阈值选择，因此结果属于内部验证结果，可能偏乐观，不能解释为独立外部验证性能。患者级概率可用于排序和探索性分析，不应脱离校准结果直接作为临床决策概率。

## 影像组学流程

按以下顺序运行：

```powershell
python Code\Radiomics_feature_extraction.py --workers 4 --checkpoint-every 5
python Code\Radiomics_feature_screening.py
python Code\Radiomics_Machin_Learning_Modeling.py
```

特征提取以患者文件夹编号为准回查临床表，使用 GTVnx 优先级规则选择 ROI，并先裁剪到 ROI 三维外接框。预处理影像为 (HU+1000)/1400 的 [0,1] 缩放，箱宽默认取 25 HU 的等价缩放值（25/1400），可通过 `--bin-width`、`--log-sigmas` 调整。启用原始图像、Wavelet 和 LoG（sigma=1、2、3），提取 1130 个特征。特征筛选在每个训练交叉验证折内完成 Z-score、Pearson |r|>0.9、MUSE Top30（`--top-k`、`--first-stage`、`--muse-p` 可调）和 LASSO 五折交叉验证（λ 由五折交叉验证选择）；建模用贝叶斯优化（Optuna TPE，`--tuning bayes`，默认 30 试验/模型）为每个模型选超参数；最终验证集仅应用训练集全量拟合的筛选器，完全不参与筛选。

探索性对比脚本（调参只用训练集交叉验证，验证集只评估一次；产物统一写入 `outputs/exploratory/`）：

```powershell
python Code\Radiomics_config_sweep.py      # 多种筛选配置 × 6 模型 + 集成的对比扫描
python Code\Radiomics_variant_compare.py   # 合并各特征变体（箱宽 / LoG）的扫描结果
python Code\Radiomics_lambda_curve.py      # 固定超参数的 LASSO 惩罚强度（lambda）敏感性曲线
python Code\Radiomics_ttest_lasso.py       # t 检验 + LASSO 筛选对照
python Code\Radiomics_rfe.py               # 递归特征消除（RFE/RFECV）对照
python Code\Radiomics_muse_lasso.py        # MUSE + LASSO 筛选对照
python Code\Radiomics_muse_p_sensitivity.py # MUSE 论文参数 p 的敏感性对照（固定超参数口径）
python Code\Radiomics_joint_model.py       # 临床变量 + Rad-score 联合模型对比
python Code\Model_auc_pairwise_test.py     # 临床 vs 影像验证集 AUC 配对检验（DeLong + bootstrap，只读已固定产物）
```

- 全部特征：`Radiomics_features.csv`
- 筛选结果：`outputs/final/01_临床筛选/`
- 模型评估：`outputs/final/02_模型评估/`
- 患者级 Rad-score 与预测：`outputs/final/03_患者级结果/`
