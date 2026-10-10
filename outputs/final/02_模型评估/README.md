# 影像组学模型评估

比较模型：LR、SVM、Random Forest、LightGBM、XGBoost、ExtraTrees。

所有调参只在训练集五折交叉验证内完成，并在每个训练折内重新执行完整特征筛选；验证集只进行最终评估。评价指标包括 ROC-AUC、PR-AUC、灵敏度、特异度、准确率和 Brier score。当前验证阈值由训练集嵌套交叉验证预测的 Youden 指数确定。特征提取箱宽已校正为 25 HU 等价（`binWidth = 25/1400`，见 `Code/Radiomics_feature_extraction.py`）。

## 正式流程（嵌套筛选口径）

筛选链为 MUSE Top30 → LASSO(CV-λ)；模型超参数用 **Optuna TPE 贝叶斯优化**在每个训练折内选择（`build_bayes_specs()`，每个模型 30 试验，seed=42，搜索空间在上一轮网格端点外放宽；目标函数为训练集五折嵌套 CV AUC，每折内重跑完整筛选）。原网格搜索保留为 `--tuning grid` 对照（`build_grid_specs()`，50 个组合）。

主要文件：`radiomics_model_comparison.csv`、`radiomics_validation_metrics.csv`、`radiomics_model_summary.json`（`tuning: bayesian_optuna`）、`best_radiomics_model.pkl`。

当前正式结果（验证集 81 例，评估一次；入选规则 = 训练集嵌套 CV AUC 降序）：**ExtraTrees 入选**，训练 CV 0.6632、验证 AUC 0.6719、PR-AUC 0.7372、Brier 0.217；RandomForest 验证 AUC 0.6765、SVM 0.6671、LR 0.6503、LightGBM 0.6335、XGBoost 0.5755。验证集灵敏度/特异度/准确率使用训练 OOF Youden 阈值 0.5821（AUC 与 PR-AUC 为阈值无关的排序指标）。对比表 `radiomics_model_comparison.csv` 各行的灵敏度/特异度按阈值 0.5 计算（行内 `threshold` 列自述），正式报值用 0.5821。

临床 SVM 与影像 ExtraTrees 的验证集配对检验（`model_auc_pairwise_test.csv`，`Code/Model_auc_pairwise_test.py`；回顾性描述性比较、非预注册；两侧口径不对等：临床为验证集 6 选 1 的选择后值，影像为训练集嵌套无选择值；CSV 差值列方向为影像−临床）：差值 0.0074（临床−影像），DeLong p=0.940，配对 bootstrap 95% CI [−0.205, +0.184]——未检出统计学显著差异；CI 较宽，不表述为"等效"。

### MUSE 参数 p

`Radiomics_feature_screening.py` 的 MUSE 实现支持论文参数 p（最小熵箱集累计样本质量阈值）；`--muse-p 1.0`（默认）即完整条件熵 H(y|feature) 简化。p ∈ {0.2, 0.5, 1.0} 的两类对照（均在 `../../exploratory/`）：

- `radiomics_muse_p_sensitivity.csv`：固定超参数、特征一次性拟合的快速对照（非嵌套：特征选择未按折重做，OOF 含选择泄漏），与嵌套协议数字不可直接比较，方向不一定偏高；仅看趋势。
- `muse_p_compare/`：p=0.5 的完整嵌套跑（贝叶斯调参）。最佳训练 CV 0.6295（XGBoost），未超过切换门槛（p=1.0 最佳 0.6632 + 0.01），正式流程维持 p=1.0。该目录不覆盖正式产物。

## 筛选方式与特征变体对比（探索性，防泄露口径；产物在 `../../exploratory/`）

- `radiomics_config_sweep.csv` / `_selected.csv` / `_summary.json`：8 个筛选配置 × 6 模型 + 集成（主特征表），按跨折合并 OOF AUC 选择。
- `radiomics_variant_comparison.csv` / `_summary.json`：4 个特征变体（标准 25 HU + LoG σ1-3 / 箱宽 10 HU / 箱宽 50 HU / LoG σ1-5）× 8 筛选配置 × 7 模型共 224 组合的合并表。`best_by_validation_optimistic` 字段是按验证集取最大值的乐观口径；该口径下影像侧为 224 选 1，与临床模型的 6 选 1 选择压力不同，引用时需注意。
- `radiomics_lambda_curve.csv`：固定超参数下 LASSO 惩罚 C ∈ {0.1, 0.2, 0.3, 0.5, 1.0} 的敏感性曲线（快速对照，只变特征选择）。
- `radiomics_ttest_lasso.csv`：t 检验（训练集 CR vs non-CR，等方差）按 p<0.05 / p<0.01 过滤后接 LASSO 的对照（同样为固定超参数的快速对照）。
- `radiomics_rfe.csv`：递归特征消除对照（RFECV-LR、RFECV-ExtraTrees、固定 30 个的 RFE）。
- `radiomics_muse_lasso.csv`：MUSE（Minimum Uncertainty and Sample Elimination，IEEE TKDE 2019）× LASSO 对照（MUSE Top30/Top15 → LASSO 五折 CV 选 λ），同口径下验证 AUC 高于 mRMR+LASSO 对照。

## 联合模型（临床 + 影像）

`joint_model_comparison.csv`、`joint_model_summary.json`：临床 3 变量与 Rad-score 的同口径对比（临床 / 影像 / 联合三个特征集）。

## 临床模型

`validation_model_comparison.csv`、`best_model_holdout_metrics.csv`、`clinical_best_model.json`、`best_clinical_model.pkl`。

## 旧版产物（已归档）

`radiomics_fixed_feature_*`、`radiomics_fixed_feature_retuned_*`、`calibration_curve.png`（早于最新临床重跑）已移至 `../../archive/pre_binwidth_fix/`。

轮次 2 的网格搜索口径完整产物（`radiomics_model_comparison.csv` 等 5 个文件，RF 0.6536/0.6765 为网格最佳）归档于 `../../archive/round2_grid_tuning/`，供贝叶斯对照复算。
