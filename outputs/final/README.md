# 正式产出（final）

本目录为项目的正式产出，论文/汇报引用这里的文件。探索性对照在 `../exploratory/`，旧版在 `../archive/`。

## 01_临床筛选

临床侧：

- `selected_predictors.json`：入选的 3 个临床变量。
- `univariable_logistic.csv`：单因素 Logistic 回归。
- `multivariable_logistic.csv`：多因素 Logistic 回归。
- `vif.csv`：共线性评估。
- `clinical_screened_dataset.csv`：筛选后的临床建模数据集。

影像侧：

- `screening_dataset.csv`：影像建模数据集（268 例 × 1130 特征 + Rad-score）。
- `selected_features.json`：最终 5 个特征（含 LASSO 系数与标准化参数）。
- `screening_summary.json`：筛选链统计（1130→335→30→5）。
- `radiomics_rad_score.csv`：268 例 Rad-score。

## 02_模型评估

临床：

- `validation_model_comparison.csv`：6 个模型在合并验证集上的比较。
- `best_model_holdout_metrics.csv`：最佳模型（SVM）最终验证结果。
- `clinical_best_model.json`：最佳模型、变量和验证集阈值。
- `best_clinical_model.pkl`：保存的最佳临床模型。
- `calibration_curve.csv`：校准曲线数据。

影像：

- `radiomics_model_comparison.csv`：6 个模型正式对比表（训练 CV / 验证 AUC）。
- `radiomics_validation_metrics.csv`：入选模型（ExtraTrees）验证集详值。
- `radiomics_model_summary.json`：正式运行元数据（best_model / tuning / n_trials / muse_p / threshold）。
- `best_radiomics_model.pkl`：保存的入选影像模型。

分析：

- `joint_model_comparison.csv` / `joint_model_summary.json`：临床 / 影像 / 联合三特征集对比。
- `model_auc_pairwise_test.csv`：临床 vs 影像验证集 AUC 配对检验（DeLong + bootstrap）。

## 03_患者级结果

- `clinical_patient_predictions.csv`：临床最佳模型的每位患者 CR 概率。
- `radiomics_patient_predictions.csv`：影像入选模型的患者级预测（含 Rad-score、阈值 0.5821）。
- `joint_model_patient_predictions.csv`：联合模型的患者级预测。
- `training_leave_one_out_influence.csv`：临床训练集留一法影响分析。
- `radiomics_extraction_failures.csv`：影像特征提取失败记录（本次 268/268 全部成功）。

当前流程：训练集拟合 + 嵌套筛选，验证集只评估一次；没有独立测试集。
