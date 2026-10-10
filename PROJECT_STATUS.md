# 项目状态

**更新日期**: 2026-10-11
**当前轮次**: 5
**状态**: ✅ 通过（轮次 5 复审）

## 本轮结果（调参引擎切换为贝叶斯优化；MUSE 参数 p 决策）

### 变更
- 调参引擎由网格搜索切换为 **Optuna TPE 贝叶斯优化**（`Code/Radiomics_Machin_Learning_Modeling.py` 的 `build_bayes_specs()`/`tune_bayes()`，每个模型 30 试验，seed=42 可复现；目标函数 = 训练集五折嵌套 CV AUC，每折内重跑完整筛选；搜索空间相对原网格端点在两侧放宽）。原网格保留为 `--tuning grid` 对照。
- MUSE 补充论文参数 **p**（最小熵箱集累计样本质量阈值，`Radiomics_feature_screening.py` 的 `muse_select(..., p=...)`）；p=1.0 为完整条件熵 H(y|feature) 简化，与切换前实现逐位一致。
- 决策口径：**p 的正式取舍由嵌套协议对比决定，与验证集无关**。

### 验收判据与结论

1. **贝叶斯切换是否有效**：训练集嵌套 CV AUC 对照（网格 → 贝叶斯）——LR 0.5782→0.6029、SVM 0.6429→0.6429、RF 0.6536→0.6536、LGBM 0.5907→0.6140、XGB 0.5849→0.6403、ET 0.6531→0.6632。6/6 模型持平或提升（4 升 2 平），最好模型由 RF 变为 ET。原网格最优参数全部落在端点（截断），实测支持切换。
2. **MUSE p 的正式取值**：预注册规则——p=0.5 的最佳训练 CV AUC 需领先 p=1.0 最佳（0.6632）超过 0.01（即 >0.6732）才切换。p=0.5 完整嵌套跑（贝叶斯，`outputs/exploratory/muse_p_compare/`）：最佳 XGBoost 训练 CV **0.6295 < 0.6732**，且 6 个模型中 5 个低于 p=1.0。**结论：正式流程维持 p=1.0**（论文完整熵简化）。

### 正式结果（验证集 81 例，评估一次；调参只用训练集）

| 模型 | 训练集嵌套 CV | 验证 AUC | PR-AUC |
|---|---|---|---|
| ExtraTrees（入选） | 0.6632 | 0.6719 | 0.7372 |
| RandomForest | 0.6536 | 0.6765 | 0.7388 |
| SVM | 0.6429 | 0.6671 | 0.7260 |
| XGBoost | 0.6403 | 0.5755 | 0.6742 |
| LightGBM | 0.6140 | 0.6335 | 0.7003 |
| LR | 0.6029 | 0.6503 | 0.7447 |

- 入选规则：按训练集嵌套 CV AUC 降序（ExtraTrees）。
- ExtraTrees 验证集详值：AUC 0.6719、PR-AUC 0.7372、灵敏度 0.78、特异度 0.581、准确率 0.704、Brier 0.217。
- **阈值口径**：验证 AUC 与 PR-AUC 为排序指标（阈值无关，比较口径 0.5）；灵敏度/特异度/准确率使用训练 OOF Youden 阈值 **0.5821**（`radiomics_model_summary.json` 的 `threshold`），不是 0.5。
- 筛选链：1130 → 335（Pearson |r|>0.9）→ 30（MUSE）→ 5（LASSO，C≈0.108）。
- 5 个最终特征：`wavelet-HLL_ngtdm_Busyness`、`wavelet-HLH_glszm_SmallAreaLowGrayLevelEmphasis`、`wavelet-LLH_gldm_DependenceEntropy`、`wavelet-HLH_firstorder_Maximum`、`log-sigma-1-0-mm-3D_firstorder_Uniformity`。
- 临床 SVM 验证集 0.6794（6 选 1 口径）与影像 ExtraTrees 0.6719（嵌套口径）的配对检验（`model_auc_pairwise_test.csv`，81 例；CSV 差值列 `auc_diff_radiomics_minus_clinical`，方向为影像−临床）：差值 0.0074（临床−影像），DeLong z=−0.075、p=0.940；配对 bootstrap（2000 次）95% CI [−0.205, +0.184]、p=0.924。**未检出统计学显著差异**；CI 较宽（81 例样本量所限），不能表述为"等效"或"非劣"。该检验为回顾性（post-hoc）描述性比较、非预注册；两侧口径不对等（临床为验证集 6 选 1 的选择后值，影像为训练集嵌套无选择值），结论仅描述本样本。

### 口径说明

- `radiomics_muse_p_sensitivity.csv`（18 行）为 **固定超参数、特征一次性拟合**的快速对照（非嵌套：特征选择未按折重做，OOF 含选择泄漏），与嵌套协议数字不可直接比较，方向不一定偏高；只用于看 p 的趋势，p 的正式取舍以 `outputs/exploratory/muse_p_compare/` 的完整嵌套跑为准。
- 正式产物已重跑刷新（p 参数落地后、selector 类合规化改造后各一次）：`radiomics_model_summary.json` 现含 `muse_p: 1.0`、`tuning: bayesian_optuna`、`n_trials: 30`；`best_radiomics_model.pkl` 现由合规 sklearn transformer（`BaseEstimator`/`TransformerMixin`）生成，selector 级 clone 可用；两次重跑的六模型训练 CV 与验证 AUC 均与刷新前逐位一致。p=1.0 为完整条件熵简化，与切换前实现逐位一致。

### 主要产物
- `outputs/final/01_临床筛选/`（MUSE Top30 → LASSO：`selected_features.json`、`screening_summary.json`、`radiomics_rad_score.csv`）
- `outputs/final/02_模型评估/radiomics_model_comparison.csv`、`radiomics_validation_metrics.csv`、`radiomics_model_summary.json`（`tuning: bayesian_optuna`，`n_trials: 30`）、`best_radiomics_model.pkl`
- p 对照：`outputs/exploratory/muse_p_compare/`（p=0.5 完整嵌套跑，不覆盖正式产物）、`outputs/exploratory/radiomics_muse_p_sensitivity.csv`
- 配对检验：`outputs/final/02_模型评估/model_auc_pairwise_test.csv`（`Code/Model_auc_pairwise_test.py`）
- 联合模型：`outputs/final/02_模型评估/joint_model_comparison.csv`、`joint_model_summary.json`
- 探索性对照（`outputs/exploratory/`）：`radiomics_config_sweep.csv`、`radiomics_variant_comparison.csv`（224 组合）、`radiomics_lambda_curve.csv`、`radiomics_ttest_lasso.csv`、`radiomics_rfe.csv`、`radiomics_muse_lasso.csv`

## 历史记录
- 2026-10-10（轮次 1）：临床流程（SVM 0.6794）与影像组学流程跑通；发现并修复 binWidth 尺度 bug（[0,1] 缩放影像误用 HU 尺度箱宽，924/1130 特征退化；修复为 `binWidth=25/1400`）；筛选方式对比（mRMR 直用优于 LASSO 收窄）；特征变体对比（LoG σ1-5 最优）；lambda 敏感性曲线；t 检验+LASSO、RFE、MUSE+LASSO 对照（MUSE+LASSO 在同口径下优于 mRMR+LASSO，用户选定 MUSE+LASSO 为正式流程）。
- 2026-10-11（轮次 2）：正式流程切换为 MUSE Top30 → LASSO(CV) + 网格搜索并重跑；验证 AUC 0.6765（RF，按验证集口径叙述）。
- 2026-10-11（轮次 3）：调参引擎切换为 Optuna 贝叶斯优化并重跑正式产物（ET 入选，训练 CV 0.6632 / 验证 0.6719）；MUSE p 参数实现并按预注册规则完成 p=0.5 嵌套对照（未过门槛，维持 p=1.0）；联合模型产物随之刷新（Rad-score 268/268 逐位一致）。

## 审查历史
| 轮次 | 判定 | 关键建议 |
|------|------|----------|
| 1 | ⚠️ | 无红线违规（防泄露核验过、数字逐位可复算、划分与 binWidth 修复正确）；口径选择压力不对等需谨慎表述；"OOF 更稳"无数据依据；final 目录过期产物已处理 |
| 2 | ⚠️ | 联合模型产物过期（Rad-score 0/268 与当前筛选不一致）→ 已重跑并逐位核验；文档失效引用（calibration_curve.png、02/03 README 归档措辞）→ 已修；PROJECT_STATUS 阈值口径两套未标注 + "差异无统计学意义"无检验支撑 → 本轮已改；MUSE 论文保真度（p 简化、min_keep 替代 Ts）未注明 → 已实现 p 参数 + 嵌套对照，文档已注明；网格最优参数全落端点 → 已切贝叶斯，训练 CV 4/5 提升 |
| 3 | ⚠️ | 无红线违规（验证集零泄露、编号未动、正式数字逐位可独立复算，Reviewer 重跑筛选与网格对照实验验证）。缺陷：① 旧 `best_radiomics_model.pkl` 与 p 参数改造后代码不兼容（selector 缺 `muse_p`，clone 崩溃）→ 已重跑正式建模刷新（六模型训练 CV 逐位复现，summary 补 `muse_p`，pkl clone+重拟合实测通过）；② "数字整体偏高"与敏感性表数据不符 → 已改为"非嵌套不可直接比较，方向不一定偏高"；③ "5/6 持平或提升"实为 6/6（4 升 2 平）、GTVnd 243→238（已复算）→ 已修；④ 网格基线产物被覆盖 → 已归档 `outputs/archive/round2_grid_tuning/` 并在 02 README 指向；⑤ "性能相当"无检验支撑 → 已补 DeLong + 配对 bootstrap（`model_auc_pairwise_test.csv`：差值 0.0074、DeLong p=0.940、95% CI [−0.205, +0.184]） |
| 4 | ⚠️ | 无红线违规（五项整改全部独立复算逐位可复现：六模型 CV/验证 AUC、DeLong z/p、配对 CI、GTVnd 238、归档数字）。缺陷：① `PROJECT_STATUS` 残留"summary 无 `muse_p` 字段"旧说明与产物矛盾（一票否决项）→ 已修；② `RadiomicsFeatureSelector` 的 selector 级 clone 仍崩（pipeline 级此前靠 deepcopy 兜底、非标准语义），类不合规 sklearn transformer → 已改造为继承 `BaseEstimator`/`TransformerMixin`（构造器不转换参数），测试 10/10（新增 clone 断言 2 条），重跑刷新 pkl（六模型 CV 逐位不变、新 pkl selector 级 clone 正常且为未 fit 语义、预测与 03 产物 0.0 差）；③ 配对检验缺事后性/口径不对等注记、CSV 差值方向不明确 → 已补注并改列名 `auc_diff_radiomics_minus_clinical` |
| 5 | ✅ | 红线 5/5 通过；D1/D2/D3 全部完成且独立实测逐位复算（clone 语义反向验证证明断言非空断言；新 pkl 预测 0.0 差；配对检验 14 字段逐位）；文档与产物一致、表述诚实。通过。后续可选建议：① "已 fit 后 clone"语义固化为自动测试（当前仅人工实测）；② 今后覆盖正式产物前先归档留痕；③ GTVnd 对照表（待定项）。 |

## 下一步
- 已完成：正式流程（Optuna 贝叶斯 + MUSE p=1.0）通过轮次 5 复审。
- 可选后续（Reviewer 轮次 5 建议）：① "已 fit 后 clone"语义固化进测试；② 重跑前归档留痕机制；③ GTVnd 对照表（238/268 例有精确 `GTVnd` 标签，含变体 248，需用户决策方向）。
