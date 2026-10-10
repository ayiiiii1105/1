# 患者级结果

影像：`radiomics_patient_predictions.csv` 保留原始患者编号，包含数据集分组、真实标签、Rad-score、CR 概率、预测标签和阈值。

临床：`clinical_patient_predictions.csv`（最佳临床模型 CR 概率与临床评分）；联合模型见 `joint_model_patient_predictions.csv`。

影像特征提取失败记录位于 `radiomics_extraction_failures.csv`。本次 ROI 外接框裁剪模式下训练集和验证集共 268 例均成功提取。

`training_leave_one_out_influence.csv` 为临床模型的训练集留一法影响分析。影像组学留一法系列文件（`radiomics_loo_auc_*`）为旧版（binWidth 修复前的 7 特征版本），已归档至 `../../archive/pre_binwidth_fix/`。
