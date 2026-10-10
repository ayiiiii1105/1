"""Tune six radiomics models on train and evaluate once on validation."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from Radiomics_feature_screening import RadiomicsFeatureSelector, _feature_columns


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


def metrics(y_true, probability, threshold=0.5):
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, probability)), "pr_auc": float(average_precision_score(y_true, probability)), "brier_score": float(brier_score_loss(y_true, probability)), "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"), "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"), "accuracy": float(accuracy_score(y_true, prediction)), "threshold": float(threshold)}


def build_model_specs():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LR": (Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]), {"model__C": np.logspace(-4, 3, 12), "model__solver": ["liblinear", "lbfgs"]}),
        "SVM": (Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=42))]), {"model__C": np.logspace(-3, 2, 10), "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto"]}),
        "RandomForest": (RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1), {"n_estimators": [200, 400, 800], "max_depth": [None, 3, 5, 8], "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", "log2", 0.5, 1.0]}),
        "LightGBM": (LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1), {"n_estimators": [50, 100, 200, 400], "learning_rate": [0.01, 0.03, 0.1], "num_leaves": [3, 7, 15, 31], "min_child_samples": [5, 10, 20], "reg_lambda": [0, 1, 10]}),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=1), {"n_estimators": [50, 100, 200, 400], "max_depth": [1, 2, 3, 5], "learning_rate": [0.01, 0.03, 0.1], "min_child_weight": [1, 3, 8], "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0], "reg_lambda": [0, 1, 10]}),
        "ExtraTrees": (ExtraTreesClassifier(random_state=42, class_weight="balanced", n_jobs=1), {"n_estimators": [200, 400, 800], "max_depth": [None, 3, 5, 8], "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", "log2", 0.5, 1.0]}),
    }


def build_grid_specs():
    """正式流程使用的紧凑网格（网格搜索）：参数范围取自此前随机搜索的优胜区间。"""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LR": (Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]), {"model__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], "model__solver": ["liblinear"]}),
        "SVM": (Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=42))]), {"model__C": [0.003, 0.03, 0.3, 3.0], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale"]}),
        "RandomForest": (RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1), {"n_estimators": [300, 800], "max_depth": [3, 5, None], "min_samples_leaf": [1, 4], "max_features": ["sqrt"]}),
        "LightGBM": (LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1), {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "num_leaves": [3, 15], "min_child_samples": [20], "reg_lambda": [1]}),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=1), {"n_estimators": [100, 300], "max_depth": [1, 3], "learning_rate": [0.03], "min_child_weight": [3], "subsample": [0.7], "colsample_bytree": [0.7], "reg_lambda": [1]}),
        "ExtraTrees": (ExtraTreesClassifier(random_state=42, class_weight="balanced", n_jobs=1), {"n_estimators": [300, 800], "max_depth": [3, 5, None], "min_samples_leaf": [1, 4], "max_features": ["sqrt"]}),
    }


def build_bayes_specs():
    """正式流程使用的贝叶斯优化空间（Optuna TPE）：范围在上一轮网格端点外放宽，避免端点截断。"""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    def lr_space(trial):
        return {"model__C": trial.suggest_float("model__C", 1e-4, 10.0, log=True)}

    def svm_space(trial):
        return {
            "model__C": trial.suggest_float("model__C", 1e-3, 100.0, log=True),
            "model__kernel": trial.suggest_categorical("model__kernel", ["linear", "rbf"]),
        }

    def forest_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1600, step=200),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    def lgbm_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
        }

    def xgb_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
        }

    return {
        "LR": (Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))]), lr_space),
        "SVM": (Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=42))]), svm_space),
        "RandomForest": (RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1), forest_space),
        "LightGBM": (LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1), lgbm_space),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=1), xgb_space),
        "ExtraTrees": (ExtraTreesClassifier(random_state=42, class_weight="balanced", n_jobs=1), forest_space),
    }


def tune_bayes(estimator, suggest_fn, feature_names, top_k, x_train, y_train, cv, n_trials, seed=42, muse_p=1.0):
    """Optuna TPE 贝叶斯优化；目标为训练集五折 CV AUC（每折内重跑完整筛选），验证集不参与。"""
    import optuna
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        candidate = clone(estimator).set_params(**suggest_fn(trial))
        pipeline = Pipeline([("screen", RadiomicsFeatureSelector(feature_names, top_k=top_k, muse_p=muse_p)), ("model", candidate)])
        return float(cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = Pipeline([("screen", RadiomicsFeatureSelector(feature_names, top_k=top_k, muse_p=muse_p)), ("model", clone(estimator).set_params(**study.best_params))])
    best.fit(x_train, y_train)
    return best, dict(study.best_params), float(study.best_value)


def tune_one(engine, spec, feature_names, top_k, x_train, y_train, cv, n_trials, muse_p=1.0):
    """按引擎调参：目标是训练集五折 CV AUC，验证集不参与。"""
    from sklearn.pipeline import Pipeline

    if engine == "bayes":
        estimator, suggest_fn = spec
        return tune_bayes(estimator, suggest_fn, feature_names, top_k, x_train, y_train, cv, n_trials, muse_p=muse_p)
    from sklearn.model_selection import GridSearchCV

    estimator, grid = spec
    nested_estimator = Pipeline([("screen", RadiomicsFeatureSelector(feature_names, top_k=top_k, muse_p=muse_p)), ("model", estimator)])
    nested_grid = {f"model__{key}": value for key, value in grid.items()}
    search = GridSearchCV(nested_estimator, nested_grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def run(args):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rows = read_csv(args.screening_dataset)
    selected = json.loads(args.selected_features.read_text(encoding="utf-8"))["final_features"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    feature_names = _feature_columns(train)
    x_train = np.array([[float(row.get(name, "nan")) for name in feature_names] for row in train])
    x_validation = np.array([[float(row.get(name, "nan")) for name in feature_names] for row in validation])
    y_train = np.array([int(row["y"]) for row in train])
    y_validation = np.array([int(row["y"]) for row in validation])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    specs = build_bayes_specs() if args.tuning == "bayes" else build_grid_specs()
    fitted, comparison, validation_probabilities = {}, [], {}
    for name, spec in specs.items():
        tuned, best_params, best_score = tune_one(
            args.tuning, spec, feature_names, args.top_k, x_train, y_train, cv, args.n_trials, muse_p=args.muse_p
        )
        probability = tuned.predict_proba(x_validation)[:, 1]
        fitted[name] = tuned
        validation_probabilities[name] = probability
        comparison.append({"model": name, "train_nested_cv_roc_auc": best_score, **metrics(y_validation, probability), "best_params": json.dumps(best_params, ensure_ascii=False)})
        print(f"模型 {name} 完成，训练集 CV AUC={best_score:.4f}")
    comparison.sort(key=lambda row: row["train_nested_cv_roc_auc"], reverse=True)
    best_name = comparison[0]["model"]
    best = fitted[best_name]
    train_oof = cross_val_predict(best, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, thresholds = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_train, train_oof)
    valid = np.isfinite(thresholds)
    threshold = float(thresholds[valid][np.argmax(tpr[valid] - fpr[valid])])
    validation_probability = validation_probabilities[best_name]
    holdout = {"model": best_name, "split": "validation", **metrics(y_validation, validation_probability, threshold)}
    patient_rows = []
    for row, probability in zip(validation, validation_probability):
        patient_rows.append({"编号": row["编号"], "split": "validation", "y_true": int(row["y"]), "Rad-score": float(row["Rad-score"]), "CR_probability": float(probability), "predicted_y": int(probability >= threshold), "threshold": threshold})
    for row, probability in zip(train, best.predict_proba(x_train)[:, 1]):
        patient_rows.append({"编号": row["编号"], "split": "train", "y_true": int(row["y"]), "Rad-score": float(row["Rad-score"]), "CR_probability": float(probability), "predicted_y": int(probability >= threshold), "threshold": threshold})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.patient_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "radiomics_model_comparison.csv", comparison)
    write_csv(args.output_dir / "radiomics_validation_metrics.csv", [holdout])
    write_csv(args.patient_output_dir / "radiomics_patient_predictions.csv", patient_rows)
    with (args.output_dir / "best_radiomics_model.pkl").open("wb") as handle:
        pickle.dump(best, handle)
    summary = {"best_model": best_name, "features": selected, "top_k": args.top_k, "all_features": len(feature_names), "first_stage": "muse", "muse_p": args.muse_p, "tuning": "bayesian_optuna" if args.tuning == "bayes" else "grid_search", "nested_screening": True, "threshold": threshold, "train_n": len(train), "validation_n": len(validation), "outcome": "Evaluation=1 -> y=1; Evaluation=2 -> y=0", "comparison": comparison}
    if args.tuning == "bayes":
        summary["n_trials"] = args.n_trials
    (args.output_dir / "radiomics_model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"建模完成：最佳模型 {best_name}，验证集结果输出至 {args.output_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-dataset", type=Path, default=root / "outputs/final/01_临床筛选/screening_dataset.csv")
    parser.add_argument("--selected-features", type=Path, default=root / "outputs/final/01_临床筛选/selected_features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/final/02_模型评估")
    parser.add_argument("--patient-output-dir", type=Path, default=root / "outputs/final/03_患者级结果")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--tuning", choices=["bayes", "grid"], default="bayes",
                        help="超参数调优引擎：bayes=Optuna TPE 贝叶斯优化（正式流程），grid=网格搜索（对照）")
    parser.add_argument("--n-trials", type=int, default=30, help="贝叶斯优化的试验次数（每个模型）")
    parser.add_argument("--muse-p", type=float, default=1.0, help="MUSE 最小熵箱集累计样本质量阈值（论文参数 p）")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
