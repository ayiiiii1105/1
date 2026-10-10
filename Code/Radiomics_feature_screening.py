"""训练集内嵌套特征筛选，并生成 Rad-score。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def normalize_patient_id(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def correlate_with_outcome(feature, outcome) -> float:
    feature = np.asarray(feature, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if np.std(feature) == 0 or np.std(outcome) == 0:
        return 0.0
    return float(np.corrcoef(feature, outcome)[0, 1])


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clinical_map(workbook):
    from openpyxl import load_workbook

    sheet = load_workbook(workbook, data_only=True).active
    headers = [cell.value for cell in sheet[1]]
    index = {str(name): pos + 1 for pos, name in enumerate(headers) if name is not None}
    result = {}
    for row in range(2, sheet.max_row + 1):
        patient_id = normalize_patient_id(sheet.cell(row, index["编号"]).value)
        evaluation = normalize_patient_id(sheet.cell(row, index["Evaluation"]).value)
        if evaluation in {"1", "2"}:
            result[patient_id] = {"Evaluation": evaluation, "y": 1 if evaluation == "1" else 0}
    return result


def folder_split_map(preprocess_root):
    result = {}
    for split in ("train", "validation"):
        split_dir = preprocess_root / split
        for patient_dir in split_dir.iterdir():
            if not patient_dir.is_dir() or not patient_dir.name.isdigit():
                continue
            if patient_dir.name in result:
                raise ValueError(f"患者重复出现在多个数据集: {patient_dir.name}")
            result[patient_dir.name] = split
    return result


def _feature_columns(rows):
    metadata = {"编号", "split", "status", "selected_roi", "roi_label_id", "selection_category", "Evaluation", "y", "Rad-score"}
    columns = []
    for name in rows[0]:
        if name not in metadata:
            try:
                float(rows[0][name])
            except (TypeError, ValueError):
                continue
            columns.append(name)
    return columns


def _mrmr(x, y, names, top_k):
    from sklearn.feature_selection import mutual_info_classif

    relevance = mutual_info_classif(x, y, random_state=42)
    corr = np.atleast_2d(np.abs(np.corrcoef(x, rowvar=False)))
    selected = []
    remaining = list(range(x.shape[1]))
    while remaining and len(selected) < min(top_k, x.shape[1]):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(remaining, key=lambda i: relevance[i] - float(np.mean(corr[i, selected])))
        selected.append(best)
        remaining.remove(best)
    return [names[i] for i in selected]


def _entropy(q):
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return float(-(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q)))


def _bin_stats(col, y, n_bins, p=1.0):
    """每个分位箱的（掩膜、纯度、样本数）以及熵分数；常数特征返回 None。

    熵分数为 MUSE 论文 J 的取值：按箱熵升序累计样本质量至 p（p=1 即完整条件熵 H(y|feature)），
    对累计覆盖的箱加权求和。
    """
    edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return None
    bins = np.digitize(col, edges[1:-1], right=True)
    total = len(col)
    entries = []
    stats = []
    for b in range(int(bins.max()) + 1):
        mask = bins == b
        count = int(mask.sum())
        if count == 0:
            stats.append((mask, 0.0, 0))
            continue
        p1 = float(y[mask].mean())
        entropy = _entropy(p1)
        weight = count / total
        entries.append((entropy, weight))
        stats.append((mask, min(p1, 1.0 - p1), count))
    if p >= 1.0:
        score = float(sum(weight * entropy for entropy, weight in entries))
    else:
        score = 0.0
        mass = 0.0
        for entropy, weight in sorted(entries):
            score += weight * entropy
            mass += weight
            if mass >= p:
                break
    return score, stats


def muse_select(x, y, names, top_k, n_bins=10, impurity_threshold=0.1, min_keep=50, p=1.0):
    """MUSE（Minimum Uncertainty and Sample Elimination，IEEE TKDE 2019）第一阶段筛选。

    每步在仍存活的样本上选择使类别条件熵最小的特征，并剔除该特征下近乎纯的箱中
    的样本，使后续步骤聚焦于难分样本。只使用传入折的样本拟合，返回按选择顺序排列的特征名。
    p 为论文中最小熵箱集的累计样本质量阈值（正式流程取 1.0，即完整条件熵）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    alive = np.ones(len(y), dtype=bool)
    remaining = list(range(x.shape[1]))
    selected = []
    while remaining and len(selected) < min(top_k, x.shape[1]):
        best = None
        for j in remaining:
            result = _bin_stats(x[alive, j], y[alive], n_bins, p)
            if result is None:
                continue
            score, stats = result
            if best is None or score < best[0]:
                best = (score, j, stats)
        if best is None:
            break
        _, j, stats = best
        selected.append(names[j])
        remaining.remove(j)
        keep_mask = np.zeros(int(alive.sum()), dtype=bool)
        for mask, impurity, count in stats:
            if count > 0 and impurity >= impurity_threshold:
                keep_mask |= mask
        candidate = alive.copy()
        candidate[alive] = keep_mask
        if candidate.sum() >= min_keep:
            alive = candidate
    return selected


def fit_feature_screening(x, y, names, top_k=15, pearson_threshold=0.9, cv_splits=5, first_stage="muse", muse_p=1.0):
    """只使用传入折的样本拟合：缺失值、Z-score、Pearson、MUSE（或 mRMR）、LASSO。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    names = list(names)
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)
    variance_mask = np.var(x_scaled, axis=0) > 0
    kept_names = [name for name, keep in zip(names, variance_mask) if keep]
    x_scaled = x_scaled[:, variance_mask]
    if not kept_names:
        raise ValueError("当前训练折没有可用的非恒定特征")
    corr = np.atleast_2d(np.abs(np.corrcoef(x_scaled, rowvar=False)))
    keep_indices = []
    for index in range(x_scaled.shape[1]):
        if not any(corr[index, prior] > pearson_threshold for prior in keep_indices):
            keep_indices.append(index)
    pearson_names = [kept_names[i] for i in keep_indices]
    pearson_x = x_scaled[:, keep_indices]
    if first_stage == "muse":
        stage1_names = muse_select(pearson_x, y, pearson_names, top_k, p=muse_p)
    elif first_stage == "mrmr":
        stage1_names = _mrmr(pearson_x, y, pearson_names, top_k)
    else:
        raise ValueError(f"未知的第一阶段筛选方式: {first_stage}")
    stage1_indices = [pearson_names.index(name) for name in stage1_names]
    stage1_x = pearson_x[:, stage1_indices]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    lasso = LogisticRegressionCV(Cs=30, cv=cv, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=3000, random_state=42)
    lasso.fit(stage1_x, y)
    coefficients = lasso.coef_[0]
    selected_mask = np.abs(coefficients) > 1e-12
    final_names = [name for name, keep in zip(stage1_names, selected_mask) if keep]
    final_coefficients = coefficients[selected_mask]
    if not final_names:
        final_names = stage1_names
        final_coefficients = np.zeros(len(final_names), dtype=float)
    selected_indices = [stage1_names.index(name) for name in final_names]
    return {"imputer": imputer, "scaler": scaler, "names": names, "pearson_names": pearson_names, "stage1": first_stage, "stage1_names": stage1_names, "final_features": final_names, "coefficients": final_coefficients, "lasso_C": float(lasso.C_[0]), "x_selected": stage1_x[:, selected_indices]}


class RadiomicsFeatureSelector(BaseEstimator, TransformerMixin):
    """sklearn transformer：每个CV训练折独立筛选特征。"""

    def __init__(self, feature_names, top_k=15, first_stage="muse", muse_p=1.0):
        # 合规 sklearn estimator：构造器只赋值、不转换参数（clone 的 sanity check 要求）
        self.feature_names = feature_names
        self.top_k = top_k
        self.first_stage = first_stage
        self.muse_p = muse_p

    def fit(self, x, y):
        self.result_ = fit_feature_screening(
            x, y, self.feature_names, top_k=self.top_k, first_stage=self.first_stage, muse_p=self.muse_p
        )
        self.selected_features_ = self.result_["final_features"]
        return self

    def transform(self, x):
        result = self.result_
        x_scaled = result["scaler"].transform(result["imputer"].transform(np.asarray(x, dtype=float)))
        index_map = {name: index for index, name in enumerate(result["names"])}
        indices = [index_map[name] for name in result["final_features"]]
        return x_scaled[:, indices]


def run(args):
    rows = read_csv(args.features)
    labels = clinical_map(args.workbook)
    split_map = folder_split_map(args.preprocess_root)
    joined = []
    for row in rows:
        patient_id = normalize_patient_id(row["编号"])
        split = split_map.get(patient_id)
        if split is not None and patient_id in labels:
            joined.append({**row, **labels[patient_id], "split": split})
    train = [row for row in joined if row["split"] == "train"]
    validation = [row for row in joined if row["split"] == "validation"]
    if len({row["y"] for row in train}) < 2:
        raise ValueError("训练集有效患者必须同时包含 y=0 和 y=1")
    feature_names = _feature_columns(train)
    x_train = np.array([[float(row.get(name, "nan")) for name in feature_names] for row in train], dtype=float)
    y = np.array([row["y"] for row in train], dtype=int)
    screening = fit_feature_screening(x_train, y, feature_names, top_k=args.top_k, first_stage=args.first_stage, muse_p=args.muse_p)
    final_names = screening["final_features"]
    coefficients = screening["coefficients"]
    means = dict(zip(feature_names, screening["imputer"].statistics_))
    scales = dict(zip(feature_names, screening["scaler"].scale_))
    output_rows = []
    for row in joined:
        values = np.array([float(row.get(name, "nan")) for name in final_names], dtype=float)
        values = (np.nan_to_num(values, nan=np.array([means[name] for name in final_names])) - np.array([means[name] for name in final_names])) / np.array([scales[name] for name in final_names])
        rad_score = float(np.dot(values, coefficients))
        output_rows.append({"编号": row["编号"], "split": row["split"], "y": row["y"], "Evaluation": row["Evaluation"], "Rad-score": rad_score, **{name: row.get(name, "nan") for name in feature_names}})

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_payload = {"final_features": final_names, "stage1": args.first_stage, "stage1_names": screening["stage1_names"], "top_k": args.top_k, "pearson_retained": screening["pearson_names"], "means": means, "scales": scales, "lasso_coefficients": dict(zip(final_names, coefficients.tolist())), "lasso_C": screening["lasso_C"], "lambda": float(1.0 / screening["lasso_C"]), "nested_screening": True}
    (out_dir / "selected_features.json").write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "screening_dataset.csv", output_rows, ["编号", "split", "y", "Evaluation", "Rad-score", *feature_names])
    write_csv(out_dir / "radiomics_rad_score.csv", output_rows, ["编号", "split", "y", "Evaluation", "Rad-score"])
    summary = {"train_n": len(train), "validation_n": len(validation), "raw_features": len(feature_names), "pearson_features": len(screening["pearson_names"]), "first_stage": args.first_stage, "top_k": args.top_k, "muse_p": args.muse_p, "stage1_features": len(screening["stage1_names"]), "lasso_features": len(final_names), "nested_screening": True, "positive_label": "Evaluation=1 -> y=1", "negative_label": "Evaluation=2 -> y=0"}
    (out_dir / "screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"特征筛选完成：训练集 {len(train)} 例，验证集 {len(validation)} 例，最终特征 {len(final_names)} 个")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--workbook", type=Path, default=root / "Clinical_data_cleaning.xlsx")
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/final/01_临床筛选")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--first-stage", default="muse", choices=["muse", "mrmr"])
    parser.add_argument("--muse-p", type=float, default=1.0, help="MUSE 最小熵箱集累计样本质量阈值（论文参数 p）")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
