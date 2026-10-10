"""使用已确定的影像组学特征和已调参数，按当前文件夹划分重新评估模型。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Radiomics_Machin_Learning_Modeling import build_model_specs, metrics
from Radiomics_feature_screening import clinical_map, folder_split_map, normalize_patient_id


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    raw_rows = read_csv(args.features)
    labels = clinical_map(args.workbook)
    split_map = folder_split_map(args.preprocess_root)
    rows = []
    for row in raw_rows:
        patient_id = normalize_patient_id(row["编号"])
        if patient_id in labels and patient_id in split_map:
            rows.append({**row, **labels[patient_id], "split": split_map[patient_id]})
    selected = json.loads(args.selected_features.read_text(encoding="utf-8"))["final_features"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    x_train = np.array([[float(row[name]) for name in selected] for row in train])
    x_validation = np.array([[float(row[name]) for name in selected] for row in validation])
    y_train = np.array([int(row["y"]) for row in train])
    y_validation = np.array([int(row["y"]) for row in validation])
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    results = []
    predictions = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (estimator, grid) in build_model_specs().items():
        search = RandomizedSearchCV(estimator, grid, n_iter=24, scoring="roc_auc", cv=cv, n_jobs=1, refit=True, random_state=42)
        search.fit(x_train, y_train)
        probability = search.predict_proba(x_validation)[:, 1]
        results.append({"model": name, "feature_mode": "fixed_7_features", "feature_rescreened": False, "train_n": len(train), "validation_n": len(validation), "train_cv_roc_auc": float(search.best_score_), "best_params": json.dumps(search.best_params_, ensure_ascii=False), **metrics(y_validation, probability)})
        predictions.extend({"model": name, "编号": row["编号"], "y_true": int(row["y"]), "CR_probability": float(probability_value)} for row, probability_value in zip(validation, probability))
        print(f"{name} 完成，训练集CV AUC={search.best_score_:.4f}，验证集AUC={results[-1]['roc_auc']:.4f}")
    results.sort(key=lambda row: row["roc_auc"], reverse=True)
    write_csv(args.output_dir / "radiomics_fixed_feature_retuned_model_comparison.csv", results)
    write_csv(args.output_dir / "radiomics_fixed_feature_retuned_validation_predictions.csv", predictions)
    (args.output_dir / "radiomics_fixed_feature_retuned_model_summary.json").write_text(json.dumps({"features": selected, "feature_count": len(selected), "feature_rescreened": False, "hyperparameters_retuned": True, "train_n": len(train), "validation_n": len(validation), "comparison": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"固定特征影像组学模型结果已写入：{args.output_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--selected-features", type=Path, default=root / "outputs/final/01_临床筛选/selected_features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/archive/pre_binwidth_fix")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
