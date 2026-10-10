"""合并各特征提取变体的配置扫描结果，输出跨变体比较表。

所有变体的配置均由训练集五折交叉验证调参，验证集对每个配置只评估一次；
比较表按训练集交叉验证 AUC 降序排列，另给出按验证集 AUC 的最大值供参考（同样偏乐观）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "feature_variant", "screen", "screen_description", "top_k", "pearson_threshold", "lasso_C", "model",
    "n_features", "train_cv_roc_auc", "oof_auc",
    "roc_auc", "pr_auc", "brier_score", "sensitivity", "specificity", "accuracy",
    "threshold", "best_params",
]
WINNER_FIELDS = ("feature_variant", "top_k", "model", "train_cv_roc_auc", "roc_auc")


def main():
    root = Path(__file__).resolve().parent.parent
    variants_dir = root / "outputs/archive/feature_variants"
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=[
        root / "outputs/exploratory/radiomics_config_sweep_summary.json",
        variants_dir / "hu10" / "radiomics_config_sweep_summary.json",
        variants_dir / "hu50" / "radiomics_config_sweep_summary.json",
        variants_dir / "log5" / "radiomics_config_sweep_summary.json",
    ])
    parser.add_argument("--variant-names", nargs="+", default=[
        "hu25_log123", "hu10_log123", "hu50_log123", "hu25_log12345",
    ])
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/exploratory")
    args = parser.parse_args()

    rows = []
    for path, name in zip(args.inputs, args.variant_names):
        if not path.exists():
            print(f"跳过缺失文件: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["all_configurations"]:
            rows.append({"feature_variant": name, **row})
    if not rows:
        raise ValueError("没有任何可合并的扫描结果")
    rows.sort(key=lambda row: row["oof_auc"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "radiomics_variant_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    best_by_oof = rows[0]
    best_by_fold_mean = max(rows, key=lambda row: row["train_cv_roc_auc"])
    best_by_validation = max(rows, key=lambda row: row["roc_auc"])
    summary = {
        "protocol": "所有变体的配置均由训练集五折交叉验证调参，验证集对每个配置只评估一次。"
                    "主选择口径为跨折合并的 OOF AUC（oof_auc）；折平均 CV AUC（train_cv_roc_auc）"
                    "在强正则化模型上会因跨折尺度漂移而虚高，仅作参考。",
        "n_configurations": len(rows),
        "best_by_oof_auc": {key: best_by_oof[key] for key in WINNER_FIELDS},
        "best_by_fold_mean_cv_auc": {key: best_by_fold_mean[key] for key in WINNER_FIELDS},
        "best_by_validation_optimistic": {key: best_by_validation[key] for key in WINNER_FIELDS},
        "caveat": "按验证集选择的最大值偏乐观，其真实验证性能会被高估；"
                  "临床模型的 0.679 也是在验证集上选出来的，二者口径相同。",
    }
    (args.output_dir / "radiomics_variant_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
