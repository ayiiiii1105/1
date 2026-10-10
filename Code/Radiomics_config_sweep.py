"""在训练集交叉验证上比较不同影像组学筛选/建模配置，全部配置的真实数字写入比较表。

协议：
- 特征筛选（mRMR top-K 网格 + LASSO）在完整训练集上拟合，验证集只应用筛选结果；
- 每个配置的模型超参数只在训练集五折交叉验证内调节；
- 验证集对每个配置只评估一次，任何配置不根据验证集结果修改；
- 主结果按训练集交叉验证 AUC 选择（写入 summary 的 selected 字段）。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from Radiomics_feature_screening import (
    _feature_columns,
    clinical_map,
    fit_feature_screening,
    folder_split_map,
    normalize_patient_id,
    read_csv,
    write_csv,
)
from Radiomics_Machin_Learning_Modeling import build_model_specs, metrics

SCREEN_CONFIGS = (
    ("lasso", 15, 0.90, None, "现行：mRMR Top15 → LASSO（CV 选 C）"),
    ("lasso", 30, 0.90, None, "mRMR Top30 → LASSO（CV 选 C）"),
    ("lasso", 50, 0.90, None, "mRMR Top50 → LASSO（CV 选 C）"),
    ("mrmr", 15, 0.90, None, "mRMR Top15 直接使用（不 LASSO）"),
    ("mrmr", 30, 0.90, None, "mRMR Top30 直接使用（不 LASSO）"),
    ("mrmr", 50, 0.90, None, "mRMR Top50 直接使用（不 LASSO）"),
    ("lasso", 30, 0.95, None, "Pearson 0.95 + mRMR Top30 + LASSO（CV 选 C）"),
    ("pearson_all", 15, 0.90, None, "只做 Pearson 去冗余，全部保留特征直接建模"),
    ("lasso_fixed", 0, 0.90, 0.1, "LASSO 固定 C=0.1（lambda=10）"),
    ("lasso_fixed", 0, 0.90, 0.2, "LASSO 固定 C=0.2（lambda=5）"),
    ("lasso_fixed", 0, 0.90, 0.3, "LASSO 固定 C=0.3（lambda≈3.33）"),
    ("lasso_fixed", 0, 0.90, 0.5, "LASSO 固定 C=0.5（lambda=2）"),
    ("lasso_fixed", 0, 0.90, 1.0, "LASSO 固定 C=1.0（lambda=1）"),
)
MODEL_N_ITER = 24
FIELDS = [
    "screen", "screen_description", "top_k", "pearson_threshold", "lasso_C", "model", "n_features",
    "train_cv_roc_auc", "oof_auc",
    "roc_auc", "pr_auc", "brier_score", "sensitivity", "specificity", "accuracy",
    "threshold", "best_params", "features",
]


def select_features(screening, screen_mode):
    if screen_mode == "lasso":
        return screening["final_features"]
    if screen_mode == "mrmr":
        return screening["stage1_names"]
    return screening["pearson_names"]


def fit_fixed_lasso(screening, x_all, y, feature_names, fixed_c):
    """在 Pearson 去冗余后的特征上用固定 C 的 L1 逻辑回归选特征（不按 CV 选 C）。"""
    from sklearn.linear_model import LogisticRegression

    index_map = {name: index for index, name in enumerate(feature_names)}
    scaled = screening["scaler"].transform(screening["imputer"].transform(x_all))
    indices = [index_map[name] for name in screening["pearson_names"]]
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=fixed_c, max_iter=3000, class_weight="balanced"
    )
    model.fit(scaled[:, indices], y)
    return [
        name for name, coef in zip(screening["pearson_names"], model.coef_[0])
        if abs(coef) > 1e-12
    ]


def load_joined(features_path, workbook, preprocess_root):
    rows = read_csv(features_path)
    labels = clinical_map(workbook)
    split_map = folder_split_map(preprocess_root)
    joined = []
    for row in rows:
        patient_id = normalize_patient_id(row["编号"])
        if patient_id in labels and patient_id in split_map:
            joined.append({**row, **labels[patient_id], "split": split_map[patient_id]})
    return joined


def rank_average(probability_rows):
    ranks = np.vstack([rankdata(row) for row in probability_rows])
    return ranks.mean(axis=0) / (probability_rows.shape[1] + 1)


def run(args):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

    start_time = time.perf_counter()
    print("[1/3] 读取特征表、临床标签与划分")
    joined = load_joined(args.features, args.workbook, args.preprocess_root)
    train = [row for row in joined if row["split"] == "train"]
    validation = [row for row in joined if row["split"] == "validation"]
    feature_names = _feature_columns(train)
    x_train_all = np.array(
        [[float(row.get(name, "nan")) for name in feature_names] for row in train], dtype=float
    )
    x_validation_all = np.array(
        [[float(row.get(name, "nan")) for name in feature_names] for row in validation], dtype=float
    )
    y_train = np.array([row["y"] for row in train], dtype=int)
    y_validation = np.array([row["y"] for row in validation], dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    specs = build_model_specs()
    total_steps = len(SCREEN_CONFIGS) * (len(specs) + 1)
    done = 0
    table = []
    print(f"[2/3] 训练集 {len(train)} 例、验证集 {len(validation)} 例；"
          f"共 {len(SCREEN_CONFIGS)} 个筛选配置 × {len(specs)} 个模型 + 集成，共 {total_steps} 个组合")

    for screen_mode, top_k, pearson_threshold, fixed_c, description in SCREEN_CONFIGS:
        screening = fit_feature_screening(
            x_train_all, y_train, feature_names,
            top_k=top_k if top_k else 15, pearson_threshold=pearson_threshold, first_stage="mrmr",
        )
        if screen_mode == "lasso_fixed":
            final_names = fit_fixed_lasso(screening, x_train_all, y_train, feature_names, fixed_c)
        else:
            final_names = select_features(screening, screen_mode)
        index_map = {name: index for index, name in enumerate(feature_names)}
        indices = [index_map[name] for name in final_names]
        x_train = screening["imputer"].transform(x_train_all)[:, indices]
        x_validation = screening["imputer"].transform(x_validation_all)[:, indices]
        validation_probabilities = []
        oof_probabilities = []
        for name, (estimator, grid) in specs.items():
            step_start = time.perf_counter()
            search = RandomizedSearchCV(
                estimator, grid, n_iter=MODEL_N_ITER, scoring="roc_auc", cv=cv,
                n_jobs=1, refit=True, random_state=args.random_state,
            )
            search.fit(x_train, y_train)
            probability = search.predict_proba(x_validation)[:, 1]
            oof = cross_val_predict(
                search.best_estimator_, x_train, y_train, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]
            validation_probabilities.append(probability)
            oof_probabilities.append(oof)
            table.append({
                "screen": screen_mode,
                "screen_description": description,
                "top_k": top_k,
                "pearson_threshold": pearson_threshold,
                "lasso_C": fixed_c if fixed_c is not None else (
                    screening.get("lasso_C") if screen_mode == "lasso" else ""
                ),
                "model": name,
                "n_features": len(final_names),
                "train_cv_roc_auc": float(search.best_score_),
                "oof_auc": float(roc_auc_score(y_train, oof)),
                "features": ";".join(final_names),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                **metrics(y_validation, probability),
            })
            done += 1
            elapsed = time.perf_counter() - start_time
            eta = elapsed / done * (total_steps - done)
            print(f"[{done}/{total_steps}] {screen_mode}/top_k={top_k} {name} 完成"
                  f"（特征 {len(final_names)} 个），训练CV AUC={search.best_score_:.4f}，"
                  f"验证AUC={table[-1]['roc_auc']:.4f}，"
                  f"本配置耗时 {time.perf_counter() - step_start:.1f} 秒，预计剩余 {eta / 60:.1f} 分钟")

        ensemble_validation = rank_average(np.vstack(validation_probabilities))
        ensemble_oof = rank_average(np.vstack(oof_probabilities))
        ensemble_auc = float(roc_auc_score(y_train, ensemble_oof))
        table.append({
            "screen": screen_mode,
            "screen_description": description,
            "top_k": top_k,
            "pearson_threshold": pearson_threshold,
            "lasso_C": fixed_c if fixed_c is not None else (
                screening.get("lasso_C") if screen_mode == "lasso" else ""
            ),
            "model": "Ensemble",
            "n_features": len(final_names),
            "train_cv_roc_auc": ensemble_auc,
            "oof_auc": ensemble_auc,
            "features": ";".join(final_names),
            "best_params": "rank_average_of_6_models",
            **metrics(y_validation, ensemble_validation),
        })
        done += 1
        print(f"[{done}/{total_steps}] {screen_mode}/top_k={top_k} Ensemble 完成，"
              f"训练OOF AUC={ensemble_auc:.4f}，验证AUC={table[-1]['roc_auc']:.4f}")

    winner = max(table, key=lambda row: row["oof_auc"])
    table_sorted = sorted(table, key=lambda row: row["oof_auc"], reverse=True)
    print("[3/3] 写出结果文件")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "radiomics_config_sweep.csv", table_sorted, FIELDS)
    write_csv(args.output_dir / "radiomics_config_sweep_selected.csv", [winner], FIELDS)
    (args.output_dir / "radiomics_config_sweep_summary.json").write_text(json.dumps({
        "protocol": "13 个筛选配置（CV-LASSO / mRMR 直用 / 固定 C 的 LASSO / 宽松 Pearson / 全特征）"
                    "在完整训练集上拟合；模型超参数只在训练集五折交叉验证内调节；"
                    "验证集对每个组合只评估一次；按跨折合并 OOF AUC 选择组合。",
        "screen_configs": [
            {"screen": mode, "top_k": top_k, "pearson_threshold": threshold,
             "lasso_C": fixed_c, "description": desc}
            for mode, top_k, threshold, fixed_c, desc in SCREEN_CONFIGS
        ],
        "n_iter": MODEL_N_ITER,
        "train_n": len(train),
        "validation_n": len(validation),
        "random_state": args.random_state,
        "selection": "跨折合并 OOF AUC 最高的组合",
        "selected": winner,
        "all_configurations": table_sorted,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描完成，共 {len(table)} 个组合；按 OOF AUC 选中的组合："
          f"{winner['screen']}/top_k={winner['top_k']} {winner['model']}"
          f"（特征 {winner['n_features']} 个），验证集AUC={winner['roc_auc']:.4f}")
    print(f"结果已写入：{args.output_dir}，总耗时 {time.perf_counter() - start_time:.1f} 秒")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/exploratory")
    parser.add_argument("--random-state", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
