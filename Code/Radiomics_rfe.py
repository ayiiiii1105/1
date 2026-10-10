"""递归特征消除（RFE/RFECV）筛选方式的快速对照：固定超参数，只变特征选择。

RFECV 用训练集五折交叉验证选择保留特征数（scoring=roc_auc，验证集不参与）；
另跑一个固定保留 30 个特征的 RFE 做同数对照。模型超参数固定为已完成的训练集 CV 调参结果。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
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

CONFIGS = ("RFECV-LR", "RFECV-ExtraTrees", "RFE-30-LR", "对照:mRMR Top30 直用")
CSV_FIELDS = [
    "feature_table", "config", "n_features", "model",
    "oof_auc", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score",
]


def select_features(mode, scaled_train, y_train, pearson_names, pearson_indices, cv):
    if mode == "RFECV-LR":
        selector = RFECV(
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            step=0.1, cv=cv, scoring="roc_auc", min_features_to_select=5, n_jobs=1,
        )
    elif mode == "RFECV-ExtraTrees":
        selector = RFECV(
            ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1),
            step=0.1, cv=cv, scoring="roc_auc", min_features_to_select=5, n_jobs=1,
        )
    elif mode == "RFE-30-LR":
        selector = RFE(
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            n_features_to_select=30, step=0.1,
        )
    else:
        return None, None
    x_pearson = scaled_train[:, pearson_indices]
    selector.fit(x_pearson, y_train)
    names = [name for name, keep in zip(pearson_names, selector.support_) if keep]
    hint = f"（Pearson {len(pearson_names)} → 选中 {len(names)}"
    if isinstance(selector, RFECV):
        hint += f"，CV 最优 {selector.n_features_} 个"
    hint += "）"
    return names, hint


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
    for config_name in CONFIGS:
        step_start = time.perf_counter()
        if config_name.startswith("对照"):
            names, hint = screening["stage1_names"], ""
        else:
            names, hint = select_features(
                config_name, scaled_train, y_train, pearson_names, pearson_indices, cv
            )
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
        for config_name in CONFIGS:
            sub = [r for r in all_rows if r["feature_table"] == table_name and r["config"] == config_name]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["roc_auc"])
            print(f"  {config_name:<18} 特征 {best['n_features']:>3} 个 | "
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
                        default=root / "outputs/exploratory/radiomics_rfe.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
