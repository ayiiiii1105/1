"""临床最佳模型与影像组学最佳模型的验证集 AUC 配对检验（DeLong + 配对 bootstrap）。

两侧预测都取自已固定的最终产物（验证集 81 例，各自只做一次最终评估）；
本脚本只比较两组已确定的预测概率，不重新拟合、不参与任何选择或调参。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = [
    "n_validation", "model_a", "model_b", "auc_a", "auc_b", "auc_diff_radiomics_minus_clinical",
    "delong_z", "delong_p", "bootstrap_ci_lower", "bootstrap_ci_upper",
    "bootstrap_p", "bootstrap_repeats", "random_state",
]


def read_validation(path, probability_column):
    # 临床产物沿用历史的 validation_combined 命名，影像产物为 validation
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["split"] in {"validation", "validation_combined"}
        ]
    return {row["编号"]: (int(row["y_true"]), float(row[probability_column])) for row in rows}


def midrank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(sorted_values)
    rank = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        rank[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    result = np.empty(n, dtype=float)
    result[order] = rank
    return result


def delong_pairwise(y_true, probability_a, probability_b):
    """Sun & Xu (2014) 快速 DeLong 检验，返回两组 AUC、差值的 z 与双尾 p。"""
    from scipy.stats import norm

    positive = y_true == 1
    negative = ~positive
    m = int(positive.sum())
    n = int(negative.sum())

    def placement(probability):
        pos_probability = probability[positive]
        neg_probability = probability[negative]
        tx = midrank(pos_probability)
        ty = midrank(neg_probability)
        tz = midrank(np.concatenate([pos_probability, neg_probability]))
        auc = (tz[:m].sum() / m - (m + 1) / 2) / n
        v10 = (tz[:m] - tx) / n
        v01 = 1.0 - (tz[m:] - ty) / m
        return auc, v10, v01

    auc_a, v10_a, v01_a = placement(probability_a)
    auc_b, v10_b, v01_b = placement(probability_b)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    covariance = s10 / m + s01 / n
    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    if variance <= 0:
        return float(auc_a), float(auc_b), 0.0, 1.0
    z = float((auc_a - auc_b) / np.sqrt(variance))
    return float(auc_a), float(auc_b), z, float(2.0 * (1.0 - norm.cdf(abs(z))))


def paired_bootstrap(y_true, probability_a, probability_b, repeats, rng):
    """同一索引重采样（配对），返回 AUC 差值的 95% CI 与双尾 bootstrap p。"""
    from sklearn.metrics import roc_auc_score

    n = len(y_true)
    differences = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, n, n)
        y_sample = y_true[sample]
        if y_sample.min() == y_sample.max():
            differences[index] = np.nan
            continue
        differences[index] = (
            roc_auc_score(y_sample, probability_a[sample])
            - roc_auc_score(y_sample, probability_b[sample])
        )
    valid = differences[np.isfinite(differences)]
    lower, upper = np.percentile(valid, [2.5, 97.5])
    p = 2.0 * min(float((valid <= 0).mean()), float((valid >= 0).mean()))
    return float(lower), float(upper), float(min(p, 1.0))


def run(args):
    import json

    model_a = args.model_a or "影像_" + json.loads(
        args.radiomics_summary.read_text(encoding="utf-8")
    )["best_model"]
    model_b = args.model_b or "临床_" + json.loads(
        args.clinical_best_model.read_text(encoding="utf-8")
    )["model"]
    clinical = read_validation(args.clinical_predictions, args.probability_column)
    radiomics = read_validation(args.radiomics_predictions, args.probability_column)
    mismatched = set(clinical) ^ set(radiomics)
    if mismatched:
        raise ValueError(f"两侧患者编号不一致: {sorted(mismatched)[:5]}")
    ids = sorted(clinical, key=int)
    y_true = np.array([clinical[patient][0] for patient in ids])
    if not np.array_equal(y_true, [radiomics[patient][0] for patient in ids]):
        raise ValueError("两侧 y_true 不一致")
    probability_a = np.array([radiomics[patient][1] for patient in ids])
    probability_b = np.array([clinical[patient][1] for patient in ids])
    auc_a, auc_b, z, p_delong = delong_pairwise(y_true, probability_a, probability_b)
    rng = np.random.default_rng(args.random_state)
    lower, upper, p_bootstrap = paired_bootstrap(
        y_true, probability_a, probability_b, args.bootstrap_repeats, rng
    )
    row = {
        "n_validation": len(ids),
        "model_a": model_a,
        "model_b": model_b,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff_radiomics_minus_clinical": auc_a - auc_b,
        "delong_z": z,
        "delong_p": p_delong,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_p": p_bootstrap,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_state": args.random_state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(f"配对检验（{model_a} vs {model_b}，验证集 {len(ids)} 例）：")
    print(f"  AUC {auc_a:.4f} vs {auc_b:.4f}，差值（{model_a}−{model_b}）{auc_a - auc_b:+.4f}")
    print(f"  DeLong z={z:.3f}，p={p_delong:.3f}")
    print(f"  配对 bootstrap（{args.bootstrap_repeats} 次）95% CI [{lower:+.4f}, {upper:+.4f}]，p={p_bootstrap:.3f}")
    print(f"结果已写入 {args.output}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-predictions", type=Path,
                        default=root / "outputs/final/03_患者级结果/clinical_patient_predictions.csv")
    parser.add_argument("--radiomics-predictions", type=Path,
                        default=root / "outputs/final/03_患者级结果/radiomics_patient_predictions.csv")
    parser.add_argument("--probability-column", default="CR_probability")
    parser.add_argument("--radiomics-summary", type=Path,
                        default=root / "outputs/final/02_模型评估/radiomics_model_summary.json")
    parser.add_argument("--clinical-best-model", type=Path,
                        default=root / "outputs/final/02_模型评估/clinical_best_model.json")
    parser.add_argument("--model-a", default=None, help="默认读取影像 summary JSON 的 best_model")
    parser.add_argument("--model-b", default=None, help="默认读取 clinical_best_model.json 的 model")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/final/02_模型评估/model_auc_pairwise_test.csv")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
