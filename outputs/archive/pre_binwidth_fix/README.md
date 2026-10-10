# 旧产物：binWidth 修复前

归档日期：2026-10-10

本目录文件生成于特征提取箱宽修复之前：预处理影像已缩放到 [0,1]（HU 窗 [-1000, 400]，见 `Code/Preprocess.py`），但提取仍使用 binWidth=25（HU 尺度），导致 1130 个特征中 924 个退化为常数或异常值（如 glszm_ZoneVariance 恒为 0、部分特征跨患者从 0 到 7.9e9）。

- `radiomics_fixed_feature_*`：固定 7 个旧特征的模型复算与重调参结果
- `radiomics_loo_auc_*`：基于旧 7 特征的留一法影响分析
- `calibration_curve.png`：早于 2026-10-10 21:52 临床重跑的校准图

修复后的替代产物见 `outputs/final/02_模型评估/` 与 `outputs/final/03_患者级结果/`。
