from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


MODE_ROTATE = "rotate"
MODE_EXPAND_TO_SQUARE = "expand_to_square"
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VALID_ROTATE_ANGLES = {90, 180, 270}
DEFAULT_SOURCE_IMAGE_SIZE = (160, 128)


def infer_background_fill(image: Image.Image):
    """Infer the background fill value from the four corners."""
    width, height = image.size
    corner_pixels = [
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    ]
    return Counter(corner_pixels).most_common(1)[0][0]


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """Rotate an image clockwise by 90, 180, or 270 degrees."""
    if angle not in VALID_ROTATE_ANGLES:
        raise ValueError(f"Unsupported rotation angle: {angle}. Choose from {sorted(VALID_ROTATE_ANGLES)}")

    transpose_map = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    return image.transpose(transpose_map[angle])


def expand_image_to_square(image: Image.Image, target_square_size: int) -> Image.Image:
    """Expand an image onto a square canvas using the inferred background color."""
    if target_square_size < max(image.size):
        raise ValueError(
            f"Target square size {target_square_size} is smaller than image size {image.size}"
        )

    fill_color = infer_background_fill(image)
    expanded = Image.new(image.mode, (target_square_size, target_square_size), color=fill_color)
    if image.mode == "P":
        palette = image.getpalette()
        if palette is not None:
            expanded.putpalette(palette)

    offset_x = (target_square_size - image.size[0]) // 2
    offset_y = (target_square_size - image.size[1]) // 2
    expanded.paste(image, (offset_x, offset_y))
    return expanded


def _build_summary() -> dict[str, int]:
    return {"total": 0, "processed": 0, "copied": 0}


def process_directory(
    input_dir: Path,
    output_dir: Path,
    source_size: tuple[int, int] = DEFAULT_SOURCE_IMAGE_SIZE,
    mode: str = MODE_ROTATE,
    rotate_angle: int = 90,
    target_square_size: int = 160,
) -> dict[str, int]:
    """
    Process one directory of images.

    - mode='rotate': rotate images matching source_size by rotate_angle
    - mode='expand_to_square': expand images matching source_size to target_square_size
    - all other images are copied unchanged
    """
    if mode not in {MODE_ROTATE, MODE_EXPAND_TO_SQUARE}:
        raise ValueError(f"Unsupported mode: {mode}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary()

    for image_path in sorted(path for path in input_dir.iterdir() if path.is_file()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        summary["total"] += 1
        output_path = output_dir / image_path.name

        with Image.open(image_path) as image:
            if image.size == source_size:
                if mode == MODE_ROTATE:
                    rotate_image(image, rotate_angle).save(output_path)
                else:
                    expand_image_to_square(image, target_square_size).save(output_path)
                summary["processed"] += 1
            else:
                shutil.copy2(image_path, output_path)
                summary["copied"] += 1

    return summary


def process_dataset_pair(
    input_dir: Path,
    target_dir: Path,
    output_input_dir: Path,
    output_target_dir: Path,
    source_size: tuple[int, int] = DEFAULT_SOURCE_IMAGE_SIZE,
    mode: str = MODE_ROTATE,
    rotate_angle: int = 90,
    target_square_size: int = 160,
) -> dict[str, dict[str, int]]:
    """Process both inputs and targets directories with the same configuration."""
    return {
        "inputs": process_directory(
            input_dir=input_dir,
            output_dir=output_input_dir,
            source_size=source_size,
            mode=mode,
            rotate_angle=rotate_angle,
            target_square_size=target_square_size,
        ),
        "targets": process_directory(
            input_dir=target_dir,
            output_dir=output_target_dir,
            source_size=source_size,
            mode=mode,
            rotate_angle=rotate_angle,
            target_square_size=target_square_size,
        ),
    }


def print_summary(name: str, directory: Path, output_directory: Path, summary: dict[str, int]) -> None:
    print(f"{name} directory : {directory}")
    print(f"{name} output    : {output_directory}")
    print(
        f"{name} summary   : total {summary['total']}, "
        f"processed {summary['processed']}, copied {summary['copied']}"
    )


def main() -> None:
    # ============================================================
    # Configuration: edit these values directly instead of CLI args
    # ============================================================
    MODE = MODE_ROTATE
    INPUT_DIR = "data/inputs"
    TARGET_DIR = "data/targets"
    OUTPUT_INPUT_DIR = "data/inputs_rotated"
    OUTPUT_TARGET_DIR = "data/targets_rotated"

    SOURCE_IMAGE_SIZE = (160, 128)   # (width, height)
    ROTATE_ANGLE = 90                # 90 / 180 / 270
    TARGET_SQUARE_SIZE = 160         # e.g. 160 / 192 / 256
    # ============================================================

    project_root = Path(__file__).resolve().parent
    input_dir = (project_root / INPUT_DIR).resolve()
    target_dir = (project_root / TARGET_DIR).resolve()
    output_input_dir = (project_root / OUTPUT_INPUT_DIR).resolve()
    output_target_dir = (project_root / OUTPUT_TARGET_DIR).resolve()

    summary = process_dataset_pair(
        input_dir=input_dir,
        target_dir=target_dir,
        output_input_dir=output_input_dir,
        output_target_dir=output_target_dir,
        source_size=SOURCE_IMAGE_SIZE,
        mode=MODE,
        rotate_angle=ROTATE_ANGLE,
        target_square_size=TARGET_SQUARE_SIZE,
    )

    print(f"mode            : {MODE}")
    print(f"source size     : {SOURCE_IMAGE_SIZE[0]} x {SOURCE_IMAGE_SIZE[1]}")
    if MODE == MODE_ROTATE:
        print(f"rotate angle    : {ROTATE_ANGLE}")
    else:
        print(f"square size     : {TARGET_SQUARE_SIZE}")
    print_summary("inputs", input_dir, output_input_dir, summary["inputs"])
    print_summary("targets", target_dir, output_target_dir, summary["targets"])


if __name__ == "__main__":
    main()
