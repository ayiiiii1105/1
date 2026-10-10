"""MUSE（Minimum Uncertainty and Sample Elimination）+ LASSO 筛选方式的快速对照。

MUSE（IEEE TKDE 2019，二分类增量特征选择）：每步在仍存活的样本上，
选择使类别条件熵 H(y|feature) 最小的特征（等价于互信息最大）；
然后剔除该特征下近乎纯的箱中的样本（"样本消除"），使后续步骤聚焦于难分样本。

对照口径：MUSE 选择前 m 个特征（可与 mRMR Top30 直接比较），再进一步用
LASSO（五折 CV 选 lambda）收窄。模型超参数固定为已完成的训练集 CV 调参结果。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from Radiomics_feature_screening import (
    _feature_columns,
    clinical_map,
    fit_feature_screening,
    folder_split_map,
    muse_select,
    normalize_patient_id,
    read_csv,
)
from Radiomics_lambda_curve import FAMILIES, load_fixed_params
from Radiomics_Machin_Learning_Modeling import build_model_specs, metrics

CONFIGS = [
    ("MUSE Top30→LASSO(CV)", "muse_lasso", 30),
    ("MUSE Top30→直接建模", "muse_only", 30),
    ("MUSE Top15→LASSO(CV)", "muse_lasso", 15),
    ("对照:mRMR Top30→LASSO(CV)", "mrmr_lasso", 30),
]
CSV_FIELDS = [
    "feature_table", "config", "n_features", "model",
    "oof_auc", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score",
]


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
    index_map = {name: i for i, name in enumerate(feature_names)}
    scaled_train = screening["scaler"].transform(screening["imputer"].transform(x_train))
    imputed_train = screening["imputer"].transform(x_train)
    imputed_validation = screening["imputer"].transform(x_validation)
    pearson_names = screening["pearson_names"]
    pearson_indices = [index_map[n] for n in pearson_names]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    muse_names = muse_select(scaled_train[:, pearson_indices], y_train, pearson_names, 30)
    results = []
    for config_name, mode, top_m in CONFIGS:
        step_start = time.perf_counter()
        if mode == "muse_lasso":
            chosen = muse_names[:top_m]
            x_sel = scaled_train[:, [index_map[n] for n in chosen]]
            lasso = LogisticRegressionCV(
                Cs=30, cv=cv, penalty="l1", solver="liblinear",
                scoring="roc_auc", max_iter=3000, random_state=42,
            )
            lasso.fit(x_sel, y_train)
            coef = lasso.coef_[0]
            names = [n for n, c in zip(chosen, coef) if abs(c) > 1e-12]
            hint = f"（MUSE {len(chosen)} → LASSO 后 {len(names)}，C={lasso.C_[0]:.4g}）"
        elif mode == "muse_only":
            names = muse_names[:top_m]
            hint = ""
        else:
            chosen = screening["stage1_names"][:top_m]
            x_sel = scaled_train[:, [index_map[n] for n in chosen]]
            lasso = LogisticRegressionCV(
                Cs=30, cv=cv, penalty="l1", solver="liblinear",
                scoring="roc_auc", max_iter=3000, random_state=42,
            )
            lasso.fit(x_sel, y_train)
            coef = lasso.coef_[0]
            names = [n for n, c in zip(chosen, coef) if abs(c) > 1e-12]
            hint = f"（mRMR {len(chosen)} → LASSO 后 {len(names)}，C={lasso.C_[0]:.4g}）"
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
        print(f"[{table_name}] {config_name}{hint}: 最终 {len(names)} 个特征，"
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
                        default=root / "outputs/exploratory/radiomics_muse_lasso.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
