# 影像组学特征筛选

- 结局：`Evaluation=1` 为 CR，`y=1`；`Evaluation=2` 为 non-CR，`y=0`。
- 数据规模：训练集 187 例，验证集 81 例。
- 流程：每个模型交叉验证训练折内独立执行 Z-score → Pearson 绝对相关系数大于 0.9 去冗余 → MUSE Top 30（Minimum Uncertainty and Sample Elimination，IEEE TKDE 2019；论文参数 p=1.0，即完整条件熵简化，`--muse-p` 可调）→ LASSO Logistic 五折交叉验证（λ 由五折交叉验证选择；嵌套特征筛选）。
- 当前原始特征：1130 个；Pearson 后 335 个；MUSE 30 个；LASSO 最终 5 个（C≈0.108，即 λ≈9.24，由五折 CV 选出）。
- 验证集只应用训练集全量拟合的筛选器和参数，不参与筛选。
- MUSE 参数 p 的取舍（p ∈ {0.2, 0.5, 1.0}）：预注册规则为 p=0.5 的完整嵌套最佳训练 CV 需超 p=1.0 最佳（0.6632）0.01 才切换；实测 0.6295 未过门槛，维持 p=1.0。对照产物见 `../../exploratory/muse_p_compare/` 与 `../../exploratory/radiomics_muse_p_sensitivity.csv`。
- 特征提取箱宽已按 [0,1] 缩放影像校正为 25 HU 等价（`binWidth = 25/1400`，见 `Code/Radiomics_feature_extraction.py` 的 `DEFAULT_BIN_WIDTH`）；对比实验见 `../../exploratory/radiomics_config_sweep.csv` 与 `../../exploratory/radiomics_variant_comparison.csv`。

临床侧文件（同目录）：`selected_predictors.json`、`univariable_logistic.csv`、`multivariable_logistic.csv`、`vif.csv`、`clinical_screened_dataset.csv`。

影像侧主要文件：`selected_features.json`、`screening_dataset.csv`、`radiomics_rad_score.csv`、`screening_summary.json`。
