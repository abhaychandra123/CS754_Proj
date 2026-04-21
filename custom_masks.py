"""
Custom mask generation utilities for Phase 3 inpainting experiments.

This module can be imported by the main pipeline, and can also be run as a
standalone script to generate and preview masks.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import data as skdata
from skimage.transform import resize


DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_MASK_MODE = "random"
DEFAULT_MISSING_FRAC = 0.30
DEFAULT_SEED = 42
DEFAULT_SQUARE_SIZE = 20
DEFAULT_SQUARE_CENTER = (44, 84)
DEFAULT_SCRATCH_COUNT = 3
DEFAULT_SCRATCH_THICKNESS = 4
DEFAULT_OVERLAY_TEXT = "SAMPLE"
DEFAULT_TEXT_SCALE = 0.24
DEFAULT_TEXT_ANGLE = -90.0
DEFAULT_TEXT_STROKE = 0.5

MASK_MODES = (
    "random",
    "face-square",
    "scratches",
    "face-square+scratches",
    "text-overlay",
)


def random_dropout_mask(
    shape: tuple[int, int],
    missing_frac: float = DEFAULT_MISSING_FRAC,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Create a random binary mask where 0 indicates missing pixels."""
    if not 0.0 <= missing_frac <= 1.0:
        raise ValueError("missing_frac must be in [0, 1].")

    rng = np.random.default_rng(seed)
    return (rng.uniform(size=shape) >= missing_frac).astype(np.float64)


def square_hole_mask(
    shape: tuple[int, int],
    square_size: int = DEFAULT_SQUARE_SIZE,
    center: tuple[int, int] | None = DEFAULT_SQUARE_CENTER,
) -> np.ndarray:
    """Create a binary mask with a solid square hole (all zeros) in it."""
    h, w = shape
    size = int(max(1, min(square_size, h, w)))
    if center is None:
        center = (h // 2, w // 2)

    cy, cx = int(center[0]), int(center[1])
    y0 = max(0, cy - size // 2)
    y1 = min(h, y0 + size)
    x0 = max(0, cx - size // 2)
    x1 = min(w, x0 + size)

    mask = np.ones(shape, dtype=np.float64)
    mask[y0:y1, x0:x1] = 0.0
    return mask


def diagonal_scratches_mask(
    shape: tuple[int, int],
    scratch_count: int = DEFAULT_SCRATCH_COUNT,
    thickness: int = DEFAULT_SCRATCH_THICKNESS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Create a binary mask with thick random diagonal scratches."""
    h, w = shape
    yy, xx = np.indices(shape)
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=np.float64)

    thickness = max(1, int(thickness))
    scratch_count = max(1, int(scratch_count))

    for _ in range(scratch_count):
        # Draw line segments that span left-to-right with random diagonal tilt.
        x0, x1 = 0, w - 1
        y0 = int(rng.integers(-h // 5, h + h // 5))

        if rng.uniform() < 0.5:
            y1 = int(y0 + h + rng.integers(-h // 5, h // 5 + 1))
        else:
            y1 = int(y0 - h + rng.integers(-h // 5, h // 5 + 1))

        denom = np.hypot(y1 - y0, x1 - x0)
        if denom == 0:
            continue

        # Distance-from-line threshold gives a thick scratch.
        dist = np.abs((y1 - y0) * xx - (x1 - x0) * yy + x1 * y0 - y1 * x0) / denom
        mask[dist <= (thickness / 2.0)] = 0.0

    return mask


def _load_bold_font(font_size: int):
    """Best-effort load of a bold TrueType font, with safe fallback."""
    from PIL import ImageFont

    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def text_overlay_mask(
    shape: tuple[int, int],
    text: str = DEFAULT_OVERLAY_TEXT,
    text_scale: float = DEFAULT_TEXT_SCALE,
    text_angle: float = DEFAULT_TEXT_ANGLE,
    text_stroke: int = DEFAULT_TEXT_STROKE,
) -> np.ndarray:
    """Create a mask where rendered white text pixels are marked missing."""
    if not text or not text.strip():
        raise ValueError("overlay text must be a non-empty string.")

    from PIL import Image, ImageDraw

    h, w = shape
    base_size = max(8, int(min(h, w) * max(0.05, float(text_scale))))
    stroke_width = max(0, int(text_stroke))

    # Render text on a black canvas so white text pixels can be mapped to holes.
    canvas = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(canvas)

    font_size = base_size
    font = _load_bold_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Fit text width to the image while preserving readability.
    while (text_w > int(0.95 * w) or text_h > int(0.90 * h)) and font_size > 8:
        font_size -= 1
        font = _load_bold_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2 - bbox[0]
    y = (h - text_h) // 2 - bbox[1]
    draw.text(
        (x, y),
        text,
        fill=255,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=255,
    )

    if float(text_angle) != 0.0:
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        canvas = canvas.rotate(
            float(text_angle),
            resample=resample,
            expand=False,
            fillcolor=0,
        )

    text_pixels = np.array(canvas, dtype=np.uint8)
    mask = np.ones(shape, dtype=np.float64)
    mask[text_pixels > 0] = 0.0
    return mask


def generate_mask(
    image_shape: tuple[int, int],
    mode: str = DEFAULT_MASK_MODE,
    *,
    missing_frac: float = DEFAULT_MISSING_FRAC,
    seed: int = DEFAULT_SEED,
    square_size: int = DEFAULT_SQUARE_SIZE,
    square_center: tuple[int, int] | None = DEFAULT_SQUARE_CENTER,
    scratch_count: int = DEFAULT_SCRATCH_COUNT,
    scratch_thickness: int = DEFAULT_SCRATCH_THICKNESS,
    overlay_text: str = DEFAULT_OVERLAY_TEXT,
    text_scale: float = DEFAULT_TEXT_SCALE,
    text_angle: float = DEFAULT_TEXT_ANGLE,
    text_stroke: int = DEFAULT_TEXT_STROKE,
) -> np.ndarray:
    """
    Generate a binary mask where 1 means observed and 0 means missing.

    Modes
    -----
    random
        Independent random pixel drops controlled by missing_frac.
    face-square
        A single solid square hole (default centered near cameraman face).
    scratches
        Thick random diagonal scratches across the image.
    face-square+scratches
        Union of square hole and scratches.
    text-overlay
        White "SAMPLE TEXT" overlay where text pixels are treated as missing.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {MASK_MODES}")

    if mode == "random":
        return random_dropout_mask(image_shape, missing_frac=missing_frac, seed=seed)

    if mode == "face-square":
        return square_hole_mask(image_shape, square_size=square_size, center=square_center)

    if mode == "scratches":
        return diagonal_scratches_mask(
            image_shape,
            scratch_count=scratch_count,
            thickness=scratch_thickness,
            seed=seed,
        )

    if mode == "text-overlay":
        return text_overlay_mask(
            image_shape,
            text=overlay_text,
            text_scale=text_scale,
            text_angle=text_angle,
            text_stroke=text_stroke,
        )

    square = square_hole_mask(image_shape, square_size=square_size, center=square_center)
    scratches = diagonal_scratches_mask(
        image_shape,
        scratch_count=scratch_count,
        thickness=scratch_thickness,
        seed=seed,
    )
    return (square * scratches).astype(np.float64)


def _camera_image(image_size: tuple[int, int]) -> np.ndarray:
    """Load and resize skimage camera image to [0,1] float."""
    img_raw = skdata.brick()
    return resize(img_raw.astype(np.float64) / 255.0, image_size, anti_aliasing=True)


def _save_preview(img_clean: np.ndarray, mask: np.ndarray, save_path: str) -> None:
    """Save a quick visual preview of the generated mask."""
    img_corrupt = img_clean * mask

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_clean, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    miss_rgba = np.zeros((*mask.shape, 4), dtype=np.float64)
    miss_rgba[..., 0] = 1.0
    miss_rgba[..., 3] = (mask == 0) * 0.6
    axes[1].imshow(img_clean, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(miss_rgba)
    axes[1].set_title("Mask Overlay")
    axes[1].axis("off")

    axes[2].imshow(img_corrupt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Corrupted")
    axes[2].axis("off")

    fig.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate custom masks for the inpainting pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=MASK_MODES,
        default=DEFAULT_MASK_MODE,
        help="Mask pattern to generate.",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--missing-frac", type=float, default=DEFAULT_MISSING_FRAC)
    parser.add_argument("--square-size", type=int, default=DEFAULT_SQUARE_SIZE)
    parser.add_argument(
        "--square-center",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=DEFAULT_SQUARE_CENTER,
        help="Square center as ROW COL.",
    )
    parser.add_argument("--scratch-count", type=int, default=DEFAULT_SCRATCH_COUNT)
    parser.add_argument("--scratch-thickness", type=int, default=DEFAULT_SCRATCH_THICKNESS)
    parser.add_argument(
        "--overlay-text",
        default=DEFAULT_OVERLAY_TEXT,
        help="Text used for text-overlay mode.",
    )
    parser.add_argument(
        "--text-scale",
        type=float,
        default=DEFAULT_TEXT_SCALE,
        help="Text size as a fraction of min(image_height, image_width).",
    )
    parser.add_argument(
        "--text-angle",
        type=float,
        default=DEFAULT_TEXT_ANGLE,
        help="Rotation angle (degrees) for text-overlay mode.",
    )
    parser.add_argument(
        "--text-stroke",
        type=int,
        default=DEFAULT_TEXT_STROKE,
        help="Stroke width for bold text rendering in text-overlay mode.",
    )
    parser.add_argument(
        "--save-mask",
        default=os.path.join("outputs", "custom_mask.npy"),
        help="Path to save the generated mask as .npy.",
    )
    parser.add_argument(
        "--save-preview",
        default=os.path.join("outputs", "custom_mask_preview.png"),
        help="Path to save a visual preview PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_size = (args.height, args.width)
    square_center = tuple(args.square_center) if args.square_center else None

    mask = generate_mask(
        image_shape=image_size,
        mode=args.mode,
        missing_frac=args.missing_frac,
        seed=args.seed,
        square_size=args.square_size,
        square_center=square_center,
        scratch_count=args.scratch_count,
        scratch_thickness=args.scratch_thickness,
        overlay_text=args.overlay_text,
        text_scale=args.text_scale,
        text_angle=args.text_angle,
        text_stroke=args.text_stroke,
    )

    mask_out_dir = os.path.dirname(args.save_mask)
    if mask_out_dir:
        os.makedirs(mask_out_dir, exist_ok=True)
    np.save(args.save_mask, mask)

    img_clean = _camera_image(image_size)
    _save_preview(img_clean, mask, args.save_preview)

    missing = int((mask == 0).sum())
    total = int(mask.size)
    print(f"Mode            : {args.mode}")
    print(f"Mask shape      : {mask.shape}")
    print(f"Missing pixels  : {missing}/{total} ({(missing / total) * 100:.1f}%)")
    print(f"Mask saved      : {args.save_mask}")
    print(f"Preview saved   : {args.save_preview}")


if __name__ == "__main__":
    main()
