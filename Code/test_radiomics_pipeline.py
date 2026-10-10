import numpy as np
import SimpleITK as sitk

from Radiomics_feature_extraction import crop_to_roi, select_roi
from Radiomics_feature_screening import (
    RadiomicsFeatureSelector,
    _feature_columns,
    normalize_patient_id,
    correlate_with_outcome,
    fit_feature_screening,
)
from Radiomics_Machin_Learning_Modeling import build_model_specs


def test_select_roi_uses_confirmed_priority_and_excludes_plan():
    labels = {
        "1": "PGTVnx",
        "2": "GTVndboost",
        "3": "GTVnx-w",
        "4": "GTVnx PLAN",
    }

    assert select_roi(labels) == ("GTVnx-w", 3, "gtvnx_variant")


def test_select_roi_prefers_other_gtv_over_pgtvnx():
    labels = {"1": "PGTVnx", "2": "GTVnd", "3": "GTVp"}

    assert select_roi(labels) == ("GTVnd", 2, "other_gtv")


def test_patient_id_normalization_preserves_folder_identifier():
    assert normalize_patient_id(43.0) == "43"
    assert normalize_patient_id("043") == "043"


def test_correlation_with_outcome_returns_finite_score():
    score = correlate_with_outcome(np.array([0.0, 1.0, 2.0]), np.array([0, 0, 1]))

    assert np.isfinite(score)
    assert score > 0


def test_crop_to_roi_returns_roi_bounding_box_and_binary_mask():
    image = sitk.Image([10, 12, 14], sitk.sitkFloat32)
    mask = sitk.Image([10, 12, 14], sitk.sitkUInt8)
    mask[3, 4, 5] = 7
    mask[5, 6, 7] = 7

    cropped_image, cropped_mask = crop_to_roi(image, mask, 7)

    assert cropped_image.GetSize() == (3, 3, 3)
    assert cropped_mask.GetSize() == (3, 3, 3)
    assert int(sitk.GetArrayFromImage(cropped_mask).max()) == 1


def test_feature_columns_exclude_clinical_outcomes():
    rows = [{"编号": "1", "split": "train", "Evaluation": "1", "y": 1, "status": "ok", "original_firstorder_Mean": "0.5"}]

    assert _feature_columns(rows) == ["original_firstorder_Mean"]


def test_model_tuning_specs_cover_all_requested_models():
    specs = build_model_specs()

    assert set(specs) == {"LR", "SVM", "RandomForest", "LightGBM", "XGBoost", "ExtraTrees"}
    assert all(specs[name][1] for name in specs)


def test_feature_screening_is_fit_on_the_supplied_fold_only():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 4))
    y = np.array([0, 1] * 15)
    names = ["f1", "f2", "f3", "f4"]

    result = fit_feature_screening(x, y, names, top_k=3)

    assert set(result["final_features"]).issubset(set(names))
    assert result["x_selected"].shape[0] == len(y)


def test_feature_selector_is_cloneable_sklearn_transformer():
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    selector = RadiomicsFeatureSelector(["f1", "f2", "f3", "f4"], top_k=3)
    cloned = clone(selector)

    assert cloned.get_params()["top_k"] == 3
    assert not hasattr(cloned, "result_")

    pipeline = Pipeline([("screen", RadiomicsFeatureSelector(["f1", "f2", "f3", "f4"], top_k=3))])
    cloned_pipeline = clone(pipeline)

    assert not hasattr(cloned_pipeline.named_steps["screen"], "result_")


def test_feature_selector_clone_fit_matches_direct_fit():
    from sklearn.base import clone

    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 4))
    y = np.array([0, 1] * 15)
    names = ["f1", "f2", "f3", "f4"]

    direct = RadiomicsFeatureSelector(names, top_k=3).fit(x, y).transform(x)
    cloned = clone(RadiomicsFeatureSelector(names, top_k=3)).fit(x, y).transform(x)

    assert np.allclose(direct, cloned)
