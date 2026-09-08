"""Non-destructive image watermarking engine (Pillow-only, no Streamlit imports).

This module contains all pixel-level work for "Watermark Studio":

* ``prepare_image``          -> decode + EXIF-normalise upload bytes into RGBA
* ``apply_text_watermark``   -> single or tiled, rotated, outlined text
* ``apply_logo_watermark``   -> scaled + blended logo placement
* ``encode_png``             -> lossless PNG bytes (for preview / download)

Everything operates on copies of the input image, so the caller's image
objects are never mutated -> the watermarking step is guaranteed
non-destructive and the original file stays untouched on disk.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #

POSITIONS: tuple[str, ...] = (
    "Top Left",
    "Top Center",
    "Top Right",
    "Center Left",
    "Center",
    "Center Right",
    "Bottom Left",
    "Bottom Center",
    "Bottom Right",
    "Custom",
)

#: Anchor fractions (horizontal, vertical) for each named position.
_GRID: dict[str, tuple[float, float]] = {
    "Top Left": (0.0, 0.0),
    "Top Center": (0.5, 0.0),
    "Top Right": (1.0, 0.0),
    "Center Left": (0.0, 0.5),
    "Center": (0.5, 0.5),
    "Center Right": (1.0, 0.5),
    "Bottom Left": (0.0, 1.0),
    "Bottom Center": (0.5, 1.0),
    "Bottom Right": (1.0, 1.0),
}

_FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


@dataclass
class PreparedImage:
    """Decoded upload: a mutable-safe RGBA working copy + source metadata."""

    rgba: Image.Image
    source_format: str
    has_alpha: bool

    @property
    def width(self) -> int:
        return self.rgba.width

    @property
    def height(self) -> int:
        return self.rgba.height

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.size


# --------------------------------------------------------------------------- #
# Loading / encoding
# --------------------------------------------------------------------------- #

def prepare_image(data: bytes) -> PreparedImage:
    """Open raw upload bytes, honour EXIF rotation and return an RGBA copy.

    Transparent PNGs / paletted GIFs with transparency decode safely to RGBA,
    so downstream code never trips over alpha-less colour modes.
    """
    with Image.open(io.BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source)
        source_format = (source.format or "PNG").upper()
        has_alpha = source.mode in ("RGBA", "LA") or (
            source.mode == "P" and "transparency" in source.info
        )
        working = source.convert("RGBA")  # forces a full pixel decode
    return PreparedImage(rgba=working, source_format=source_format, has_alpha=has_alpha)


def encode_png(image: Image.Image, keep_alpha: bool = True) -> bytes:
    """Serialise an RGBA/RGB image to PNG bytes without quality loss."""
    buf = io.BytesIO()
    if keep_alpha and image.mode == "RGBA":
        out = image
    else:
        out = image.convert("RGB")
    out.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #

def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a TrueType font of the requested size (cached), with fallbacks."""
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[size] = font
            return font
        except OSError:
            continue
    try:
        font = ImageFont.load_default(size)
    except TypeError:  # very old Pillow fallback
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _alpha(opacity: float) -> int:
    """Map a 0..1 opacity onto an 8-bit alpha channel value."""
    return int(round(max(0.0, min(1.0, opacity)) * 255.0))


def _render_text_sprite(
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int],
    angle: float,
) -> Image.Image:
    """Render a single line of text onto a tight transparent sprite.

    The sprite may be rotated freely; ``expand=True`` keeps every pixel.
    """
    probe = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    except TypeError:  # bitmap default fonts ignore stroke
        bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    pad = max(4, int(round(text_h * 0.15)))

    sprite = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(sprite)
    try:
        d.text(
            (pad - bbox[0], pad - bbox[1]),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
    except TypeError:  # bitmap fonts: no stroke support
        d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill)

    if angle:
        sprite = sprite.rotate(angle, resample=Image.BICUBIC, expand=True)
    return sprite


def _place(
    position: str,
    container_w: int,
    container_h: int,
    element_w: int,
    element_h: int,
    margin: int,
    x_percent: float = 50.0,
    y_percent: float = 50.0,
) -> tuple[int, int]:
    """Top-left pixel for an element anchored inside a container.

    Named positions use the ``_GRID`` fractions; ``Custom`` maps the
    x/y percentages onto the free space so that 0% / 50% / 100% mean
    left|centre|right and top|middle|bottom respectively.
    """
    if position == "Custom":
        h_frac = max(0.0, min(1.0, float(x_percent) / 100.0))
        v_frac = max(0.0, min(1.0, float(y_percent) / 100.0))
    elif position in _GRID:
        h_frac, v_frac = _GRID[position]
    else:
        raise ValueError(f"Unknown watermark position: {position!r}")
    avail_w = max(0, container_w - element_w - 2 * margin)
    avail_h = max(0, container_h - element_h - 2 * margin)
    return (
        int(round(margin + avail_w * h_frac)),
        int(round(margin + avail_h * v_frac)),
    )


def _margin_for(image: Image.Image) -> int:
    return max(8, int(round(0.03 * min(image.size))))


# --------------------------------------------------------------------------- #
# Public watermark operations (all return a NEW image)
# --------------------------------------------------------------------------- #

def apply_text_watermark(
    image: Image.Image,
    *,
    text: str,
    font_size: int = 72,
    color: str = "#ffffff",
    opacity: float = 0.6,
    angle: int = 0,
    tile: bool = False,
    gap_percent: int = 40,
    position: str = "Bottom Right",
    x_percent: float = 50.0,
    y_percent: float = 50.0,
    outline: bool = True,
) -> Image.Image:
    """Return ``image`` with a text watermark composited on top of it."""
    image = image.convert("RGBA")
    width, height = image.size

    text = " ".join((text or "").split())  # normalise whitespace -> single line
    if not text:
        raise ValueError("Watermark text cannot be empty.")

    font_size = max(8, int(font_size))
    alpha = _alpha(opacity)
    if alpha <= 0:
        return image.copy()

    try:
        color_rgba = ImageColor.getrgb(color)  # hex like "#rrggbb" (or names)
    except ValueError as exc:
        raise ValueError(f"Invalid watermark colour: {color!r}") from exc

    font = _load_font(font_size)
    fill = (color_rgba[0], color_rgba[1], color_rgba[2], alpha)
    stroke_width = max(1, font_size // 30) if outline else 0
    stroke_fill = (0, 0, 0, alpha)
    sprite = _render_text_sprite(text, font, fill, stroke_width, stroke_fill, float(angle))
    sprite_w, sprite_h = sprite.size
    if sprite_w < 1 or sprite_h < 1:
        return image.copy()

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    if tile:
        gap = max(2, int(round(sprite_w * max(0, gap_percent) / 100.0)))
        step_x = max(1, sprite_w + gap)
        step_y = max(1, sprite_h + gap)
        y = -sprite_h
        while y < height:
            x = -sprite_w
            while x < width:
                overlay.paste(sprite, (x, y), sprite)  # sprite doubles as mask
                x += step_x
            y += step_y
    else:
        margin = _margin_for(image)
        x, y = _place(position, width, height, sprite_w, sprite_h, margin,
                      x_percent, y_percent)
        overlay.paste(sprite, (x, y), sprite)

    return Image.alpha_composite(image, overlay)


def apply_logo_watermark(
    image: Image.Image,
    logo: Image.Image,
    *,
    width_percent: float = 18.0,
    opacity: float = 0.7,
    position: str = "Bottom Right",
    x_percent: float = 50.0,
    y_percent: float = 50.0,
) -> Image.Image:
    """Return ``image`` with ``logo`` scaled, blended and composited on top.

    Works with both PNG logos (true per-pixel alpha via the mask) and
    JPEG logos (flat alpha applied from the opacity slider).
    """
    image = image.convert("RGBA")
    width, height = image.size
    logo = logo.convert("RGBA")
    logo_w, logo_h = logo.size

    if logo_w < 1 or logo_h < 1 or width_percent <= 0:
        return image.copy()

    alpha = _alpha(opacity)
    if alpha <= 0:
        return image.copy()

    # Scale relative to the base image width, preserving aspect ratio.
    target_w = max(1, int(round(width * float(width_percent) / 100.0)))
    target_h = max(1, int(round(target_w * logo_h / logo_w)))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    if alpha < 255:
        logo.putalpha(logo.getchannel("A").point(lambda v: (v * alpha) // 255))

    margin = _margin_for(image)
    x, y = _place(position, width, height, target_w, target_h, margin,
                  x_percent, y_percent)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay.paste(logo, (x, y), logo)
    return Image.alpha_composite(image, overlay)
