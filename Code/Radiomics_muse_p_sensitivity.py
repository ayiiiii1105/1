"""MUSE 论文参数 p 的敏感性对照：最小熵箱集累计样本质量阈值 p ∈ {0.2, 0.5, 1.0}。

MUSE（IEEE TKDE 2019）的 J 取"按箱熵升序累计样本质量至 p"的加权熵和；正式流程用
p=1.0（完整条件熵 H(y|feature)）简化。本脚本在固定超参数下比较 p 的三种取值：
每种 p 各取 MUSE Top30 → LASSO（五折 CV 选 lambda），验证集只评估一次。
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

P_VALUES = (0.2, 0.5, 1.0)
CSV_FIELDS = [
    "feature_table", "p", "n_features", "model",
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

    results = []
    for p_value in P_VALUES:
        step_start = time.perf_counter()
        muse_names = muse_select(scaled_train[:, pearson_indices], y_train, pearson_names, 30, p=p_value)
        x_sel = scaled_train[:, [index_map[n] for n in muse_names]]
        lasso = LogisticRegressionCV(
            Cs=30, cv=cv, penalty="l1", solver="liblinear",
            scoring="roc_auc", max_iter=3000, random_state=42,
        )
        lasso.fit(x_sel, y_train)
        coef = lasso.coef_[0]
        names = [n for n, c in zip(muse_names, coef) if abs(c) > 1e-12]
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
                "p": p_value,
                "n_features": len(names),
                "model": family,
                "oof_auc": float(roc_auc_score(y_train, oof)),
                **metrics(y_validation, probability),
            })
        print(f"[{table_name}] p={p_value}: MUSE 30 → LASSO 后 {len(names)} 个特征，"
              f"耗时 {time.perf_counter() - step_start:.1f} 秒")
    return results


def run(args):
    start_time = time.perf_counter()
    fixed_params = load_fixed_params(args.comparison_csv, args.fixed_params_variant)
    all_rows = evaluate_table(
        "hu25_主表", args.main_features, args.workbook, args.preprocess_root, fixed_params
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== 汇总（每个 p 下验证 AUC 最高的模型）===")
    for p_value in P_VALUES:
        sub = [r for r in all_rows if r["p"] == p_value]
        best = max(sub, key=lambda r: r["roc_auc"])
        print(f"  p={p_value:<4} 特征 {best['n_features']:>3} 个 | "
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
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/exploratory/radiomics_muse_p_sensitivity.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
