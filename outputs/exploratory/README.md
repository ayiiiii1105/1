# 探索性对照产物

本目录为探索性对照的产物存档（固定超参数或非正式口径），**不是正式结果**；正式结果见 `../final/`。

| 文件 | 说明 | 生产者 |
|---|---|---|
| `radiomics_config_sweep.csv` / `_selected.csv` / `_summary.json` | 筛选配置 × 6 模型 + 集成扫描 | `Code/Radiomics_config_sweep.py` |
| `radiomics_variant_comparison.csv` / `_summary.json` | 特征变体（箱宽/LoG）合并对比（224 组合） | `Code/Radiomics_variant_compare.py` |
| `radiomics_lambda_curve.csv` | LASSO 惩罚强度敏感性曲线 | `Code/Radiomics_lambda_curve.py` |
| `radiomics_ttest_lasso.csv` | t 检验 + LASSO 对照 | `Code/Radiomics_ttest_lasso.py` |
| `radiomics_rfe.csv` | 递归特征消除对照 | `Code/Radiomics_rfe.py` |
| `radiomics_muse_lasso.csv` | MUSE + LASSO 筛选对照 | `Code/Radiomics_muse_lasso.py` |
| `radiomics_muse_p_sensitivity.csv` | MUSE 参数 p 敏感性（固定超参数口径，非嵌套） | `Code/Radiomics_muse_p_sensitivity.py` |
| `muse_p_compare/` | p=0.5 完整嵌套对照跑（含 `screening_p05/` 筛选产物）；预注册裁决依据，正式流程维持 p=1.0 | `Radiomics_feature_screening.py` + `Radiomics_Machin_Learning_Modeling.py`（`--muse-p 0.5`） |

变体特征表（`Radiomics_features_bin10/bin50/hu10/hu50/log5.csv`）在 `../archive/feature_variants/`。
