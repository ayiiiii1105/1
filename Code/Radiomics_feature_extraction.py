"""Extract 3-D PyRadiomics features from the folder-defined cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


META_COLUMNS = ["编号", "selected_roi", "roi_label_id", "status"]

# 预处理影像为 (HU + 1000) / 1400 的 [0, 1] 缩放（见 Preprocess.py 的 HU 窗 [-1000, 400]），
# 因此标准 25 HU 的固定箱宽在缩放影像上等价于 25 / 1400。
DEFAULT_BIN_WIDTH = 25.0 / 1400.0


def _is_plan(name: str) -> bool:
    return "plan" in name.lower()


def _roi_rank(name: str):
    upper = name.upper().strip()
    if _is_plan(name):
        return None
    if upper == "GTVNX":
        return (0, 0)
    if re.match(r"^GTV[-_]?NX(?:[-_].*)?$", upper):
        return (1, len(name))
    if upper == "GTV":
        return (2, 0)
    if upper.startswith("GTVND"):
        return (3, len(name))
    if upper.startswith("GTVP"):
        return (4, len(name))
    if upper.startswith("GTV"):
        return (5, len(name))
    if re.match(r"^PGTV[-_]?NX(?:[-_].*)?$", upper):
        return (6, len(name))
    return None


def select_roi(labels: dict) -> tuple[str, int, str] | None:
    """Return (ROI name, integer label, selection category)."""
    candidates = []
    for raw_id, raw_name in labels.items():
        name = str(raw_name).strip()
        rank = _roi_rank(name)
        if rank is None:
            continue
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        category = (
            "gtvnx_exact" if rank[0] == 0 else
            "gtvnx_variant" if rank[0] == 1 else
            "other_gtv" if rank[0] in {2, 3, 4, 5} else
            "pgtvnx"
        )
        candidates.append((rank, name, label_id, category))
    if not candidates:
        return None
    _, name, label_id, category = min(candidates, key=lambda item: (item[0], item[2]))
    return name, label_id, category


def _check_dependencies():
    try:
        import SimpleITK  # noqa: F401
        import radiomics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("缺少 SimpleITK 或 pyradiomics，请先安装对应依赖。") from exc


def _same_geometry(image, mask) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and image.GetSpacing() == mask.GetSpacing()
        and image.GetOrigin() == mask.GetOrigin()
        and image.GetDirection() == mask.GetDirection()
    )


def crop_to_roi(image, mask, label_id):
    """Crop both images to the selected label bounding box and binarize the mask."""
    import SimpleITK as sitk

    binary_mask = sitk.Cast(sitk.Equal(mask, int(label_id)), sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(binary_mask)
    if not stats.HasLabel(1):
        raise ValueError(f"选定 ROI 掩膜为空: label={label_id}")
    x, y, z, size_x, size_y, size_z = stats.GetBoundingBox(1)
    index = [int(x), int(y), int(z)]
    size = [int(size_x), int(size_y), int(size_z)]
    return sitk.RegionOfInterest(image, size, index), sitk.RegionOfInterest(binary_mask, size, index)


def extract_patient(patient_dir: Path, split: str, extractor):
    import SimpleITK as sitk

    image_path = patient_dir / "im.nii.gz"
    mask_path = patient_dir / "structures.nii.gz"
    label_path = patient_dir / "structures_labels.json"
    if not all(path.exists() for path in (image_path, mask_path, label_path)):
        raise FileNotFoundError("缺少 im.nii.gz、structures.nii.gz 或 structures_labels.json")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    selected = select_roi(labels)
    if selected is None:
        raise ValueError("没有可用的 GTV/GTVnx/PGTVnx ROI")
    roi_name, label_id, category = selected
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if not _same_geometry(image, mask):
        raise ValueError("图像与掩膜空间几何信息不一致")
    image, mask = crop_to_roi(image, mask, label_id)
    result = extractor.execute(image, mask, label=1)
    features = {
        str(key): float(value)
        for key, value in result.items()
        if not str(key).startswith("diagnostics_")
    }
    return features, {"selected_roi": roi_name, "roi_label_id": label_id, "selection_category": category}


_WORKER_EXTRACTOR = None


def _build_extractor(bin_width=DEFAULT_BIN_WIDTH, log_sigmas=(1.0, 2.0, 3.0)):
    from radiomics import featureextractor

    settings = {
        "binWidth": bin_width,
        "normalize": False,
        "resampledPixelSpacing": None,
        "force2D": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for name in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(name)
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(log_sigmas)})
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return extractor


def _init_worker(bin_width, log_sigmas):
    global _WORKER_EXTRACTOR
    _WORKER_EXTRACTOR = _build_extractor(bin_width, log_sigmas)


def _extract_worker(job):
    split, patient_dir = job
    try:
        features, meta = extract_patient(Path(patient_dir), split, _WORKER_EXTRACTOR)
        return {"ok": True, "row": {"编号": Path(patient_dir).name, "status": "ok", **meta, **features}}
    except Exception as exc:
        return {"ok": False, "failure": {"编号": Path(patient_dir).name, "split": split, "status": "failed", "reason": str(exc)}}


def _write_checkpoint(path, rows, failures):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False), encoding="utf-8")


def run(args):
    _check_dependencies()

    rows, failures = [], []
    patient_dirs = []
    for split in ("train", "validation"):
        split_dir = args.preprocess_root / split
        patient_dirs.extend((split, d) for d in sorted(split_dir.iterdir(), key=lambda p: int(p.name)) if d.is_dir() and d.name.isdigit())
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.bin_width, tuple(args.log_sigmas)),
    ) as pool:
        futures = [pool.submit(_extract_worker, (split, str(patient_dir))) for split, patient_dir in patient_dirs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                rows.append(result["row"])
            else:
                failures.append(result["failure"])
            rows.sort(key=lambda row: int(row["编号"]))
            failures.sort(key=lambda row: (row["split"], int(row["编号"])))
            if index % args.checkpoint_every == 0 or index == len(patient_dirs):
                _write_checkpoint(args.checkpoint, rows, failures)
                elapsed = time.perf_counter() - start
                eta = elapsed / index * (len(patient_dirs) - index)
                print(f"进度：{index}/{len(patient_dirs)}；已用 {elapsed / 60:.1f} 分钟；预计剩余 {eta / 60:.1f} 分钟", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({key for row in rows for key in row if key not in META_COLUMNS + ["selection_category"]})
    fieldnames = META_COLUMNS + ["selection_category"] + feature_names
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    with args.failure_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["编号", "split", "status", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"特征提取完成：成功 {len(rows)} 例，失败 {len(failures)} 例，输出 {args.output}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-root", type=Path, default=root / "Preprocess")
    parser.add_argument("--output", type=Path, default=root / "Radiomics_features.csv")
    parser.add_argument("--failure-output", type=Path, default=root / "outputs/final/03_患者级结果/radiomics_extraction_failures.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--log-sigmas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--checkpoint", type=Path, default=root / "Radiomics_features.checkpoint.json")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
