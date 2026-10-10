# 项目文件清单

**整理日期**: 2026-10-11
**对应状态**: `PROJECT_STATUS.md` 轮次 5（✅ 通过）

> 本文档回答三个问题：代码在哪里、正式产出是哪些、中间产物哪些可清理。
> 与 `README.md`（怎么跑）和 `PROJECT_STATUS.md`（进展与审查）互补。

## 一、代码（`Code/`）

### 1. 正式管线（按运行顺序）

| 脚本 | 作用 | 产出 |
|---|---|---|
| `Data_cleaning.py` | 原始临床表清洗（公式转值、GTV 体积保留） | `Clinical_data_cleaning.xlsx` |
| `Preprocess.py` | CT 与结构标签统一体素大小 | `Preprocess/`（train/validation 患者目录） |
| `Clinical_data_screening.py` | 临床变量筛选（单因素/多因素/VIF，只用训练集） | `01_临床筛选/` 的 `selected_predictors.json`、`univariable_logistic.csv`、`multivariable_logistic.csv`、`vif.csv`、`clinical_screened_dataset.csv` |
| `Clinical_Machine_Learning_Modeling.py` | 临床 6 模型建模与验证集评估 | `02_模型评估/` 的 `validation_model_comparison.csv`、`best_model_holdout_metrics.csv`、`clinical_best_model.json`、`best_clinical_model.pkl`、`calibration_curve.csv`；`03_患者级结果/clinical_patient_predictions.csv` |
| `Radiomics_feature_extraction.py` | 1130 维影像组学特征提取（原始+Wavelet+LoG，箱宽 25 HU 等价） | `Radiomics_features.csv`；`03_患者级结果/radiomics_extraction_failures.csv` |
| `Radiomics_feature_screening.py` | 嵌套筛选（Z-score → Pearson → MUSE Top30 → LASSO CV-λ） | `01_临床筛选/` 的 `screening_dataset.csv`、`selected_features.json`、`screening_summary.json`、`radiomics_rad_score.csv` |
| `Radiomics_Machin_Learning_Modeling.py` | 影像 6 模型贝叶斯调参与验证集评估（`--tuning bayes` 正式 / `grid` 对照） | `02_模型评估/` 的 `radiomics_model_comparison.csv`、`radiomics_validation_metrics.csv`、`radiomics_model_summary.json`、`best_radiomics_model.pkl`；`03_患者级结果/radiomics_patient_predictions.csv` |

### 2. 分析与检验

| 脚本 | 作用 | 产出 |
|---|---|---|
| `Radiomics_joint_model.py` | 临床 / 影像 / 联合三特征集同口径对比 | `02_模型评估/joint_model_comparison.csv`、`joint_model_summary.json`；`03_患者级结果/joint_model_patient_predictions.csv` |
| `Model_auc_pairwise_test.py` | 临床 vs 影像验证集 AUC 配对检验（DeLong + bootstrap，只读已固定产物） | `02_模型评估/model_auc_pairwise_test.csv` |
| `Radiomics_influence_analysis.py` | 影像留一法影响分析（现存产物为旧版，已归档） | 归档于 `outputs/archive/pre_binwidth_fix/` |

### 3. 探索性对照（固定口径，不参与正式流程）

前 7 个脚本的产物在 `outputs/exploratory/`；`Radiomics_fixed_feature_modeling.py` 为旧版流程，产物已归档至 `outputs/archive/pre_binwidth_fix/`。

| 脚本 | 作用 | 产出 |
|---|---|---|
| `Radiomics_config_sweep.py` | 筛选配置 × 6 模型 + 集成扫描 | `radiomics_config_sweep*.csv/json` |
| `Radiomics_variant_compare.py` | 特征变体（箱宽/LoG）合并对比（224 组合） | `radiomics_variant_comparison.csv/_summary.json` |
| `Radiomics_lambda_curve.py` | LASSO 惩罚强度敏感性曲线 | `radiomics_lambda_curve.csv` |
| `Radiomics_ttest_lasso.py` | t 检验 + LASSO 对照 | `radiomics_ttest_lasso.csv` |
| `Radiomics_rfe.py` | 递归特征消除对照 | `radiomics_rfe.csv` |
| `Radiomics_muse_lasso.py` | MUSE + LASSO 筛选对照 | `radiomics_muse_lasso.csv` |
| `Radiomics_muse_p_sensitivity.py` | MUSE 论文参数 p 敏感性（固定超参数口径） | `radiomics_muse_p_sensitivity.csv` |
| `Radiomics_fixed_feature_modeling.py` | 固定特征与参数重估（旧版产物已归档） | 归档于 `outputs/archive/pre_binwidth_fix/` |

### 4. 测试

| 文件 | 说明 |
|---|---|
| `test_radiomics_pipeline.py` | 10 个单测（ROI 选择、编号规范化、裁剪、筛选、selector clone 语义等），`python -m pytest Code/test_radiomics_pipeline.py -q` |

## 二、正式产出（`outputs/final/`）

### 01_临床筛选

| 文件 | 说明 |
|---|---|
| `selected_predictors.json` | 临床入选变量（3 个） |
| `clinical_screened_dataset.csv` | 临床清洗后建模数据集（268 例 + 3 变量） |
| `univariable_logistic.csv` / `multivariable_logistic.csv` / `vif.csv` | 临床筛选过程记录 |
| `screening_dataset.csv` | 影像建模数据集（268 例 × 1130 特征 + Rad-score，含 split） |
| `selected_features.json` | 影像最终 5 特征 + LASSO 系数 + 标准化参数 |
| `screening_summary.json` | 筛选链统计（1130→335→30→5） |
| `radiomics_rad_score.csv` | 268 例 Rad-score（训练集全量拟合器） |

### 02_模型评估

| 文件 | 说明 |
|---|---|
| `radiomics_model_comparison.csv` | **影像 6 模型正式对比表**（训练 CV / 验证 AUC / PR-AUC / 阈值） |
| `radiomics_validation_metrics.csv` | **入选模型（ExtraTrees）验证集详值** |
| `radiomics_model_summary.json` | 正式运行元数据（best_model/tuning/n_trials/muse_p/threshold） |
| `best_radiomics_model.pkl` | 入选影像模型（合规 sklearn Pipeline） |
| `model_auc_pairwise_test.csv` | 临床 vs 影像 AUC 配对检验（DeLong + bootstrap） |
| `joint_model_comparison.csv` / `joint_model_summary.json` | 临床 / 影像 / 联合三特征集对比 |
| `validation_model_comparison.csv` / `best_model_holdout_metrics.csv` / `clinical_best_model.json` / `best_clinical_model.pkl` / `calibration_curve.csv` | 临床模型正式产物 |

### 03_患者级结果

| 文件 | 说明 |
|---|---|
| `radiomics_patient_predictions.csv` | 影像入选模型的患者级预测（268 例，Rad-score / CR 概率 / 阈值 0.5821） |
| `clinical_patient_predictions.csv` | 临床入选模型的患者级预测 |
| `joint_model_patient_predictions.csv` | 联合模型的患者级预测 |
| `training_leave_one_out_influence.csv` | 临床训练集留一法影响分析 |
| `radiomics_extraction_failures.csv` | 影像提取失败记录（本次 268/268 全部成功，仅表头） |

## 三、中间产物

### 探索性对照（`outputs/exploratory/`）
`radiomics_config_sweep*`（3 个）、`radiomics_variant_comparison*`（2 个）、`radiomics_lambda_curve.csv`、`radiomics_ttest_lasso.csv`、`radiomics_rfe.csv`、`radiomics_muse_lasso.csv`、`radiomics_muse_p_sensitivity.csv`、`muse_p_compare/`（p=0.5 完整嵌套对照，含 `screening_p05/`）。
——保留原因：文档与审查历史中的对照数字需可复算；均非正式结论。详见该目录 `README.md`。

### 归档（`outputs/archive/`）

| 目录 | 内容 |
|---|---|
| `pre_binwidth_fix/` | binWidth 修复前旧版：`radiomics_fixed_feature_*`、旧 `calibration_curve.png`、影像 LOO 系列 |
| `nested_cv_legacy/` | 临床嵌套 CV 旧版（`model_cv_scores.csv`、`nested_cv_*`、`best_clinical_model.joblib`） |
| `feature_variants/` | 5 个特征变体表（`Radiomics_features_bin10/bin50/hu10/hu50/log5.csv`）+ 各变体扫描产物 |
| `round2_grid_tuning/` | 轮次 2 网格搜索口径完整产物（5 文件，供贝叶斯对照复算） |

### 根目录数据与文档

| 文件 | 说明 |
|---|---|
| `Clinical_data.xlsx` | 原始临床表（含 GTV 体积） |
| `Clinical_data_cleaning.xlsx` | 清洗后临床表（全部脚本的输入） |
| `Radiomics_features.csv` | 主特征表（1130 特征 × 268 例） |
| `Preprocess/` | 预处理影像（`train/` 187 例 + `validation/` 81 例） |
| `Original data/` | 患者原始数据（按编号） |
| `docs/` | 工作计划与规格存档（superpowers plans/specs） |
| `AGENTS.md` / `README.md` / `PROJECT_STATUS.md` / `FILES.md` | 协作规范 / 项目说明 / 进展与审查 / 本清单 |

### 运行日志（`logs/`，22 个）

- 正式链路（7）：`extract_main.log`、`nested_muse_grid.log`、`bayes_modeling.log`、`bayes_official_rerun.log`、`bayes_official_rerun2.log`、`joint_model_refresh.log`、`modeling_nested.log`
- 探索/对照（15）：`sweep_*.log`（7 个）、`variant_extract.log`、`extract_variants.log`、`muse_p_sensitivity.log`、`muse_p05_screening.log`、`bayes_p05_modeling.log`、`nested_top30.log`、`rfe_run.log`、`archive_grid_rerun.log`

## 四、速查：汇报/论文引用哪几个文件

| 用途 | 文件 |
|---|---|
| 影像模型主结果表 | `02_模型评估/radiomics_model_comparison.csv` |
| 入选模型验证集指标 | `02_模型评估/radiomics_validation_metrics.csv` |
| 影像筛选链与 5 特征 | `01_临床筛选/selected_features.json`、`screening_summary.json` |
| 患者级预测 | `03_患者级结果/radiomics_patient_predictions.csv` |
| 临床主结果 | `02_模型评估/validation_model_comparison.csv`、`best_model_holdout_metrics.csv` |
| 临床 vs 影像统计检验 | `02_模型评估/model_auc_pairwise_test.csv` |
| p 决策依据 | `exploratory/muse_p_compare/` + `exploratory/radiomics_muse_p_sensitivity.csv` |
| 方法与口径说明 | `README.md`、`PROJECT_STATUS.md`、各目录 `README.md` |

## 五、整理记录与剩余可清理项

2026-10-11 整理执行记录：

- **删除**：`Code/__pycache__/`、`Original data/__pycache__/`、`.pytest_cache/`（运行脚本/测试时自动重建，非异常）、`Radiomics_features.checkpoint.json`（提取已完成，可重建）。
- **归位**：22 个运行日志 → `logs/`；11 个探索产物 + `muse_p_compare/` → `outputs/exploratory/`；旧口径 `clinical_analysis/model_cv_summary.csv` → `outputs/archive/nested_cv_legacy/`；`Radiomics_fixed_feature_modeling.py` 与 `Radiomics_influence_analysis.py` 的默认输出改指 `outputs/archive/pre_binwidth_fix/`（防误跑写入正式目录）。
- **改名消歧**：`screened_dataset.csv` → `clinical_screened_dataset.csv`（与影像 `screening_dataset.csv` 仅差一字母）；`patient_predictions.csv` → `clinical_patient_predictions.csv`（与 `radiomics_patient_predictions.csv` 对称）；`best_model.json` → `clinical_best_model.json`（与 `best_clinical_model.pkl` 对齐）。
- **同步**：8 个脚本的默认路径、本清单与全部 README/PROJECT_STATUS 引用。

仍可清理（未执行，确认无引用后）：

1. `logs/` 中探索日志（`sweep_*.log` 等）——正式链路日志建议保留。
2. `exploratory/muse_p_compare/best_radiomics_model.pkl`、`radiomics_patient_predictions.csv`——决策已完成，可只留 CSV/JSON 数字。
3. `outputs/archive/feature_variants/` 的断点与变体特征表——如需要复算变体对比则保留。

**git 仓库说明**：项目内 `.git` 为早期遗迹（仅 1 个提交、唯一跟踪文件 `shenjin/Preprocess.py` 已删除但其内容与 `Code/Preprocess.py` 逐字节相同）；绝大部分文件未跟踪且无 `.gitignore`。如需启用版本管理，建议先补 `.gitignore` 再整体提交（未执行）。
