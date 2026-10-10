"""留一法评估患者对影像组学AUC的影响。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Radiomics_Machin_Learning_Modeling import build_model_specs
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


def build_fixed_estimator(model_name, best_params):
    estimator = build_model_specs()[model_name][0]
    valid_params = estimator.get_params()
    mapped = {}
    for key, value in best_params.items():
        if key in valid_params:
            mapped[key] = value
        elif key.startswith("model__") and key[7:] in valid_params:
            mapped[key[7:]] = value
        else:
            raise ValueError(f"无法映射模型参数 {key} 到 {model_name}")
    estimator.set_params(**mapped)
    return estimator


def run(args):
    from sklearn.metrics import roc_auc_score

    raw_rows = read_csv(args.features)
    labels = clinical_map(args.workbook)
    split_map = folder_split_map(args.preprocess_root)
    rows = []
    for row in raw_rows:
        patient_id = normalize_patient_id(row["编号"])
        if patient_id in labels and patient_id in split_map:
            rows.append({**row, **labels[patient_id], "split": split_map[patient_id]})
    comparison = read_csv(args.comparison)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    selected = json.loads((args.selected_features).read_text(encoding="utf-8"))["final_features"]
    x_train = np.array([[float(row.get(name, "nan")) for name in selected] for row in train])
    x_validation = np.array([[float(row.get(name, "nan")) for name in selected] for row in validation])
    y_train = np.array([int(row["y"]) for row in train])
    y_validation = np.array([int(row["y"]) for row in validation])
    all_results = []

    for model_row in comparison:
        model_name = model_row["model"]
        best_params = json.loads(model_row["best_params"])
        model = build_fixed_estimator(model_name, best_params)
        model.fit(x_train, y_train)
        validation_probability = model.predict_proba(x_validation)[:, 1]
        baseline_auc = float(roc_auc_score(y_validation, validation_probability))
        for index, row in enumerate(validation):
            keep = np.ones(len(validation), dtype=bool)
            keep[index] = False
            loo_auc = float(roc_auc_score(y_validation[keep], validation_probability[keep]))
            all_results.append({"model": model_name, "dataset": "validation", "编号": row["编号"], "y": int(row["y"]), "probability": float(validation_probability[index]), "baseline_auc": baseline_auc, "loo_auc": loo_auc, "delta_auc": loo_auc - baseline_auc, "abs_delta_auc": abs(loo_auc - baseline_auc)})

        for index, row in enumerate(train):
            keep = np.ones(len(train), dtype=bool)
            keep[index] = False
            loo_model = build_fixed_estimator(model_name, best_params)
            loo_model.fit(x_train[keep], y_train[keep])
            loo_probability = loo_model.predict_proba(x_validation)[:, 1]
            loo_auc = float(roc_auc_score(y_validation, loo_probability))
            all_results.append({"model": model_name, "dataset": "train_leave_one_out", "编号": row["编号"], "y": int(row["y"]), "probability": float(model.predict_proba(x_train[index:index + 1])[:, 1][0]), "baseline_auc": baseline_auc, "loo_auc": loo_auc, "delta_auc": loo_auc - baseline_auc, "abs_delta_auc": abs(loo_auc - baseline_auc)})
            if (index + 1) % 10 == 0:
                print(f"{model_name} 训练集留一 {index + 1}/{len(train)}")

    out_dir = args.output_dir
    prefix = args.output_prefix
    write_csv(out_dir / f"{prefix}.csv", all_results)
    summary = []
    for model_name in sorted({row["model"] for row in all_results}):
        for dataset in ("train_leave_one_out", "validation"):
            subset = [row for row in all_results if row["model"] == model_name and row["dataset"] == dataset]
            subset.sort(key=lambda row: float(row["abs_delta_auc"]), reverse=True)
            for rank, row in enumerate(subset[:10], 1):
                summary.append({"model": model_name, "dataset": dataset, "rank": rank, **row})
    write_csv(out_dir / f"{prefix}_top10.csv", summary)
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps({"method": "固定7个影像特征和当前重新调出的最优超参数；训练集逐例留一重拟合，验证集逐例留一但固定完整训练模型", "features": selected, "train_n": len(train), "validation_n": len(validation), "models": sorted({row["model"] for row in all_results})}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"留一法分析完成，结果输出至 {out_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--comparison", type=Path, default=root / "outputs/final/02_模型评估/radiomics_model_comparison.csv")
    parser.add_argument("--selected-features", type=Path, default=root / "outputs/final/01_临床筛选/selected_features.json")
    parser.add_argument("--output-prefix", default="radiomics_loo_auc_influence")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/archive/pre_binwidth_fix")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
