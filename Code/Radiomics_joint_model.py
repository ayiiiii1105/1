"""临床变量与影像组学 Rad-score 联合建模：在完全相同的口径下比较 临床、影像、联合 三个特征集。

口径说明：三个特征集都使用相同的 LogisticRegression（class_weight=balanced）；正则化参数 C
和决策阈值都只在训练集内部通过五折交叉验证确定（阈值取训练集交叉验证预测的 Youden 指数）；
验证集只做一次最终评估，不参与任何拟合或选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Clinical_data_screening import CATEGORICAL_VARIABLES, NUMERIC_VARIABLES
from Clinical_Machine_Learning_Modeling import (
    bootstrap_ci,
    build_matrix,
    metrics,
    read_rows,
    select_threshold,
)

RAD_SCORE = "Rad-score"
METRIC_FIELDS = [
    "feature_set",
    "variables",
    "n_train",
    "n_validation",
    "best_C",
    "train_cv_roc_auc",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "accuracy",
    "threshold",
    "tn",
    "fp",
    "fn",
    "tp",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
    "brier_score_ci_lower",
    "brier_score_ci_upper",
    "sensitivity_ci_lower",
    "sensitivity_ci_upper",
    "specificity_ci_lower",
    "specificity_ci_upper",
    "accuracy_ci_lower",
    "accuracy_ci_upper",
]


def read_rad_scores(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["编号"]: float(row[RAD_SCORE]) for row in csv.DictReader(handle)}


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_preprocessor(variables):
    numeric_indices = [
        index for index, name in enumerate(variables)
        if name in NUMERIC_VARIABLES or name == RAD_SCORE
    ]
    categorical_indices = [
        index for index, name in enumerate(variables) if name in CATEGORICAL_VARIABLES
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_indices),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_indices),
        ],
        remainder="drop",
    )


def evaluate(feature_set, variables, train, validation, args):
    x_train = build_matrix(train, variables)
    y_train = np.array([row["y"] for row in train])
    x_validation = build_matrix(validation, variables)
    y_validation = np.array([row["y"] for row in validation])
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(variables)),
        ("model", LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear",
            random_state=args.random_state,
        )),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    search = GridSearchCV(
        pipeline, {"model__C": np.logspace(-4, 2, 13)}, scoring="roc_auc", cv=cv, n_jobs=1
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    oof_probability = cross_val_predict(
        best, x_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold = select_threshold(y_train, oof_probability)
    probability = best.predict_proba(x_validation)[:, 1]
    rng = np.random.default_rng(args.random_state)
    result = {
        "feature_set": feature_set,
        "variables": ";".join(variables),
        "n_train": len(train),
        "n_validation": len(validation),
        "best_C": float(search.best_params_["model__C"]),
        "train_cv_roc_auc": float(search.best_score_),
        **metrics(y_validation, probability, threshold),
        **bootstrap_ci(y_validation, probability, threshold, args.bootstrap_repeats, rng),
    }
    return result, best, probability


def run(args):
    start_time = time.perf_counter()
    print("[1/3] 读取临床数据与 Rad-score")
    rows = read_rows(args.workbook, args.preprocess_root)
    rad_scores = read_rad_scores(args.rad_score)
    missing = [row["编号"] for row in rows if row["编号"] not in rad_scores]
    if missing:
        raise ValueError(f"以下患者缺少 Rad-score: {missing[:5]}（共 {len(missing)} 例）")
    for row in rows:
        row[RAD_SCORE] = rad_scores[row["编号"]]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    selected = json.loads(args.selected_predictors.read_text(encoding="utf-8"))
    clinical_variables = (
        selected.get("independent_predictors") or selected.get("univariable_p_lt_0_05", [])
    )
    if not clinical_variables:
        raise ValueError("没有可用的临床变量")

    feature_sets = [
        ("临床", list(clinical_variables)),
        ("影像", [RAD_SCORE]),
        ("联合", list(clinical_variables) + [RAD_SCORE]),
    ]
    print(f"[2/3] 训练集 {len(train)} 例、验证集 {len(validation)} 例；开始拟合三个特征集")
    results = []
    models = {}
    for index, (name, variables) in enumerate(feature_sets, start=1):
        set_start = time.perf_counter()
        result, model, probability = evaluate(name, variables, train, validation, args)
        results.append(result)
        models[name] = model
        print(
            f"[特征集 {index}/3] {name} 完成，训练CV AUC={result['train_cv_roc_auc']:.4f}，"
            f"验证集AUC={result['roc_auc']:.4f}，耗时 {time.perf_counter() - set_start:.1f} 秒"
        )

    joint_name, joint_variables = feature_sets[-1]
    joint_model = models[joint_name]
    joint_threshold = results[-1]["threshold"]
    patient_rows = []
    for split_name, subset in (("train", train), ("validation", validation)):
        probability = joint_model.predict_proba(build_matrix(subset, joint_variables))[:, 1]
        for row, value in zip(subset, probability):
            patient_rows.append({
                "编号": row["编号"],
                "split": split_name,
                "y_true": row["y"],
                "Rad-score": row[RAD_SCORE],
                "joint_CR_probability": float(value),
                "decision_threshold": float(joint_threshold),
                "predicted_y_at_threshold": int(value >= joint_threshold),
            })

    print("[3/3] 写出结果文件")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.patient_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "joint_model_comparison.csv", results, METRIC_FIELDS)
    write_csv(
        args.patient_output_dir / "joint_model_patient_predictions.csv",
        patient_rows,
        ["编号", "split", "y_true", "Rad-score", "joint_CR_probability",
         "decision_threshold", "predicted_y_at_threshold"],
    )
    (args.output_dir / "joint_model_summary.json").write_text(json.dumps({
        "protocol": "三个特征集使用相同的 LogisticRegression（class_weight=balanced）；"
                    "C 由训练集五折网格搜索确定，决策阈值取训练集交叉验证预测的 Youden 指数；"
                    "验证集只做一次最终评估，不参与任何拟合或选择。",
        "clinical_variables": clinical_variables,
        "radiomics_variable": RAD_SCORE,
        "train_n": len(train),
        "validation_n": len(validation),
        "random_state": args.random_state,
        "bootstrap_repeats": args.bootstrap_repeats,
        "comparison": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"联合模型结果已写入：{args.output_dir}；患者级结果：{args.patient_output_dir}")
    print(f"总耗时: {time.perf_counter() - start_time:.1f} 秒")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--screening-dir", type=Path, default=root / "outputs/final/01_临床筛选")
    parser.add_argument("--rad-score", type=Path, default=root / "outputs/final/01_临床筛选/radiomics_rad_score.csv")
    parser.add_argument("--selected-predictors", type=Path, default=root / "outputs/final/01_临床筛选/selected_predictors.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/final/02_模型评估")
    parser.add_argument("--patient-output-dir", type=Path, default=root / "outputs/final/03_患者级结果")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
