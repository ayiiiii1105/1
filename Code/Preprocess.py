"""将患者 CT 影像和结构标签统一到目标体素大小。"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重采样 CT 影像和结构标签，并进行 HU 窗口截断与 [0, 1] 重缩放。"
    )
    parser.add_argument("input_dir", type=Path, help="患者目录的根目录")
    parser.add_argument("output_dir", type=Path, help="预处理结果输出目录")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="目标体素大小，默认 1 1 1 mm",
    )
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行处理患者数量，默认 4",
    )
    return parser.parse_args()


def make_reference(image: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    size = tuple(
        max(1, int(round(old_size * old_spacing / new_spacing)))
        for old_size, old_spacing, new_spacing in zip(
            original_size, original_spacing, spacing
        )
    )

    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return reference


def preprocess_image(
    image: sitk.Image,
    reference: sitk.Image,
    hu_min: float,
    hu_max: float,
) -> sitk.Image:
    if hu_min >= hu_max:
        raise ValueError("hu_min 必须小于 hu_max")

    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Clamp(image, sitk.sitkFloat32, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def resample_label(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        label,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label.GetPixelID(),
    )


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    spacing: tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> bool:
    image_path = patient_dir / "im.nii.gz"
    label_path = patient_dir / "structures.nii.gz"
    if not image_path.exists() or not label_path.exists():
        print(f"[跳过] {patient_dir.name}: 缺少 im.nii.gz 或 structures.nii.gz")
        return False

    try:
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        reference = make_reference(image, spacing)
        image_output = preprocess_image(image, reference, hu_min, hu_max)
        label_output = resample_label(label, reference)

        patient_output = output_dir / patient_dir.name
        patient_output.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image_output, str(patient_output / "im.nii.gz"))
        sitk.WriteImage(label_output, str(patient_output / "structures.nii.gz"))

        labels_json = patient_dir / "structures_labels.json"
        if labels_json.exists():
            shutil.copy2(labels_json, patient_output / labels_json.name)
        print(f"[完成] {patient_dir.name}")
        return True
    except Exception as exc:  # 让单个患者失败不影响其他患者
        print(f"[失败] {patient_dir.name}: {exc}")
        return False


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {args.input_dir}")
    if any(value <= 0 for value in args.spacing):
        raise SystemExit("--spacing 的三个值必须为正数")
    if args.workers <= 0:
        raise SystemExit("--workers 必须为正整数")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    process_args = [
        (
            patient_dir,
            args.output_dir,
            tuple(args.spacing),
            args.hu_min,
            args.hu_max,
        )
        for patient_dir in patient_dirs
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        completed = sum(executor.map(lambda values: process_patient(*values), process_args))
    print(f"处理完成: {completed}/{len(patient_dirs)} 个患者")


if __name__ == "__main__":
    main()
