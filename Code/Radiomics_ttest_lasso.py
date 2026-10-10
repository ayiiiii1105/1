"""t 检验 + LASSO 筛选方式的快速对照：固定超参数，只变特征选择。

流程：训练集上对每个特征做两独立样本 t 检验（CR vs non-CR，等方差），
按 p 值阈值保留显著特征，再用 LASSO 选最终特征（CV 选 C 或固定 C）。
模型超参数固定为已完成的训练集五折 CV 调参结果；验证集只做一次评估。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from Radiomics_feature_screening import (
    _feature_columns,
    clinical_map,
    fit_feature_screening,
    folder_split_map,
    normalize_patient_id,
    read_csv,
)
from Radiomics_lambda_curve import FAMILIES, load_fixed_params
from Radiomics_Machin_Learning_Modeling import build_model_specs, metrics

CONFIGS = [
    ("t检验p<0.05→LASSO(CV)", "ttest_lasso_cv", 0.05),
    ("t检验p<0.05→LASSO(C=0.2)", "ttest_lasso_fixed", 0.05),
    ("t检验p<0.01→LASSO(CV)", "ttest_lasso_cv", 0.01),
    ("t检验p<0.05→直接建模", "ttest_only", 0.05),
    ("对照:mRMR Top30 直用", "mrmr30", None),
    ("对照:现行 Pearson+mRMR15+LASSO", "current", None),
]
CSV_FIELDS = [
    "feature_table", "config", "n_features", "model",
    "oof_auc", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score",
]


def ttest_select(xtrain_scaled, y_train, feature_names, p_threshold):
    """训练集 t 检验（等方差）保留 p < 阈值 的特征。"""
    group1 = xtrain_scaled[y_train == 1]
    group0 = xtrain_scaled[y_train == 0]
    p_values = np.array([
        stats.ttest_ind(group1[:, j], group0[:, j], equal_var=True).pvalue
        for j in range(xtrain_scaled.shape[1])
    ])
    p_values = np.nan_to_num(p_values, nan=1.0)
    return [name for name, p in zip(feature_names, p_values) if p < p_threshold], p_values


def evaluate_table(table_name, features_path, workbook, preprocess_root, fixed_params):
    rows = read_csv(features_path)
    labels = clinical_map(workbook)
    split_map = folder_split_map(preprocess_root)
    joined = []
    for row in rows:
        patient_id = normalize_patient_id(row["编号"])
        if patient_id in labels and patient_id in split_map:
            joined.append({**row, **labels[patient_id], "split": split_map[patient_id]})
    train = [row for row in joined if row["split"] == "train"]
    validation = [row for row in joined if row["split"] == "validation"]
    feature_names = _feature_columns(train)
    x_train = np.array([[float(r.get(n, "nan")) for n in feature_names] for r in train], dtype=float)
    x_validation = np.array([[float(r.get(n, "nan")) for n in feature_names] for r in validation], dtype=float)
    y_train = np.array([r["y"] for r in train], dtype=int)
    y_validation = np.array([r["y"] for r in validation], dtype=int)

    screening = fit_feature_screening(x_train, y_train, feature_names, top_k=30, first_stage="mrmr")
    screen15 = fit_feature_screening(x_train, y_train, feature_names, top_k=15, first_stage="mrmr")
    index_map = {name: i for i, name in enumerate(feature_names)}
    scaled_train = screening["scaler"].transform(screening["imputer"].transform(x_train))
    imputed_train = screening["imputer"].transform(x_train)
    imputed_validation = screening["imputer"].transform(x_validation)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for config_name, mode, p_threshold in CONFIGS:
        step_start = time.perf_counter()
        if mode.startswith("ttest"):
            selected, p_values = ttest_select(scaled_train, y_train, feature_names, p_threshold)
            indices = [index_map[n] for n in selected]
            x_sel = scaled_train[:, indices]
            if mode == "ttest_lasso_cv":
                lasso = LogisticRegressionCV(
                    Cs=30, cv=cv, penalty="l1", solver="liblinear",
                    scoring="roc_auc", max_iter=3000, random_state=42,
                )
                lasso.fit(x_sel, y_train)
                coef = lasso.coef_[0]
                names = [n for n, c in zip(selected, coef) if abs(c) > 1e-12]
                extra = f"（p<{p_threshold} 留 {len(selected)}，LASSO 后 {len(names)}，C={lasso.C_[0]:.4g}）"
            elif mode == "ttest_lasso_fixed":
                lasso = LogisticRegression(
                    penalty="l1", solver="liblinear", C=0.2, max_iter=3000, class_weight="balanced"
                )
                lasso.fit(x_sel, y_train)
                coef = lasso.coef_[0]
                names = [n for n, c in zip(selected, coef) if abs(c) > 1e-12]
                extra = f"（p<{p_threshold} 留 {len(selected)}，LASSO(C=0.2) 后 {len(names)}）"
            else:
                names = selected
                extra = f"（p<{p_threshold} 留 {len(selected)}，不做 LASSO）"
        elif mode == "mrmr30":
            names = screening["stage1_names"]
            extra = ""
        else:
            names = screen15["final_features"]
            extra = ""
        indices = [index_map[n] for n in names]
        xt = imputed_train[:, indices]
        xv = imputed_validation[:, indices]
        for family in FAMILIES:
            estimator, _ = build_model_specs()[family]
            estimator.set_params(**fixed_params[family])
            estimator.fit(xt, y_train)
            probability = estimator.predict_proba(xv)[:, 1]
            oof = cross_val_predict(estimator, xt, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            results.append({
                "feature_table": table_name,
                "config": config_name,
                "n_features": len(names),
                "model": family,
                "oof_auc": float(roc_auc_score(y_train, oof)),
                **metrics(y_validation, probability),
            })
        print(f"[{table_name}] {config_name}{extra}: 最终 {len(names)} 个特征，"
              f"耗时 {time.perf_counter() - step_start:.1f} 秒")
    return results


def run(args):
    start_time = time.perf_counter()
    fixed_params = load_fixed_params(args.comparison_csv, args.fixed_params_variant)
    all_rows = []
    tables = [
        ("hu25_主表", args.main_features),
        ("log5变体", args.log5_features),
    ]
    for table_name, path in tables:
        print(f"--- 处理 {table_name}")
        all_rows.extend(evaluate_table(table_name, path, args.workbook, args.preprocess_root, fixed_params))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== 汇总（每个配置下验证 AUC 最高的模型）===")
    for table_name, _ in tables:
        print(f"\n【{table_name}】")
        for config_name, _, _ in CONFIGS:
            sub = [r for r in all_rows if r["feature_table"] == table_name and r["config"] == config_name]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["roc_auc"])
            print(f"  {config_name:<26} 特征 {best['n_features']:>3} 个 | "
                  f"{best['model']:<12} 验证AUC={best['roc_auc']:.4f} | "
                  f"OOF={best['oof_auc']:.4f} | PR={best['pr_auc']:.4f}")
    print(f"\n结果已写入 {args.output}，总耗时 {time.perf_counter() - start_time:.1f} 秒")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path,
                        default=root / "outputs/exploratory/radiomics_variant_comparison.csv")
    parser.add_argument("--fixed-params-variant", default="hu25_log123")
    parser.add_argument("--main-features", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--log5-features", type=Path,
                        default=root / "outputs/archive/feature_variants/Radiomics_features_log5.csv")
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/exploratory/radiomics_ttest_lasso.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
