"""固定超参数下的 lambda（LASSO C）敏感性曲线：只变特征选择，快速比较。

模型超参数取自已完成的训练集五折 RandomizedSearchCV 结果（各模型族 OOF 最优行），
本轮不再重新调参，使比较只反映"特征选择方式"这一个变量。
验证集只做一次评估；同时输出跨折合并 OOF AUC 供参考。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from Radiomics_config_sweep import fit_fixed_lasso
from Radiomics_feature_screening import (
    _feature_columns,
    clinical_map,
    fit_feature_screening,
    folder_split_map,
    normalize_patient_id,
    read_csv,
)
from Radiomics_Machin_Learning_Modeling import build_model_specs, metrics

FAMILIES = ("LR", "SVM", "RandomForest", "LightGBM", "XGBoost", "ExtraTrees")
CONFIGS = [
    ("lasso_C0.1", "fixed", 0.1),
    ("lasso_C0.2", "fixed", 0.2),
    ("lasso_C0.3", "fixed", 0.3),
    ("lasso_C0.5", "fixed", 0.5),
    ("lasso_C1.0", "fixed", 1.0),
    ("lasso_CV(现行)", "cv", None),
    ("mrmr_top30", "mrmr30", None),
]
CSV_FIELDS = [
    "feature_table", "config", "n_features", "model",
    "oof_auc", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score",
]


def load_fixed_params(comparison_csv, variant):
    rows = list(csv.DictReader(comparison_csv.open(encoding="utf-8-sig", newline="")))
    fixed = {}
    for family in FAMILIES:
        sub = [r for r in rows if r["feature_variant"] == variant and r["model"] == family]
        if not sub:
            continue
        best = max(sub, key=lambda r: float(r["oof_auc"]))
        fixed[family] = json.loads(best["best_params"])
    return fixed


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

    screen30 = fit_feature_screening(x_train, y_train, feature_names, top_k=30, first_stage="mrmr")
    screen15 = fit_feature_screening(x_train, y_train, feature_names, top_k=15, first_stage="mrmr")
    index_map = {name: i for i, name in enumerate(feature_names)}
    results = []
    for config_name, mode, fixed_c in CONFIGS:
        if mode == "fixed":
            names = fit_fixed_lasso(screen30, x_train, y_train, feature_names, fixed_c)
        elif mode == "mrmr30":
            names = screen30["stage1_names"]
        else:
            names = screen15["final_features"]
        indices = [index_map[n] for n in names]
        xt = screen30["imputer"].transform(x_train)[:, indices]
        xv = screen30["imputer"].transform(x_validation)[:, indices]
        model_start = time.perf_counter()
        validation_probabilities = []
        oof_probabilities = []
        for family in FAMILIES:
            estimator, _ = build_model_specs()[family]
            estimator.set_params(**fixed_params[family])
            estimator.fit(xt, y_train)
            probability = estimator.predict_proba(xv)[:, 1]
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof = cross_val_predict(estimator, xt, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            from sklearn.metrics import roc_auc_score
            validation_probabilities.append(probability)
            oof_probabilities.append(oof)
            results.append({
                "feature_table": table_name,
                "config": config_name,
                "n_features": len(names),
                "model": family,
                "oof_auc": float(roc_auc_score(y_train, oof)),
                **metrics(y_validation, probability),
            })
        print(f"[{table_name}] {config_name}: {len(names)} 个特征，"
              f"耗时 {time.perf_counter() - model_start:.1f} 秒")
    return results


def run(args):
    start_time = time.perf_counter()
    fixed_params = load_fixed_params(args.comparison_csv, args.fixed_params_variant)
    print(f"已读取 {len(fixed_params)} 个模型族的固定超参数（来源：{args.fixed_params_variant}）")
    all_rows = []
    tables = [
        ("hu25_主表", args.main_features),
        ("log5变体", args.log5_features),
    ]
    for table_name, path in tables:
        print(f"--- 处理 {table_name}: {path.name}")
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
            print(f"  {config_name:<16} 特征 {best['n_features']:>3} 个 | "
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
                        default=root / "outputs/exploratory/radiomics_lambda_curve.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
