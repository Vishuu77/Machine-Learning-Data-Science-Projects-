"""OCR core logic for the Streamlit text-extraction app.

Backend #1: easyocr 1.7.2 (CPU). Models (detector craft_mlt_25k.pth + recogniser
english_g2.pth, ~100 MB) are auto-downloaded into ~/.EasyOCR/model on first use.
Backend #2 (fallback): Windows built-in OCR (Windows.Media.Ocr) driven through
the PowerShell helper script ocr_image.ps1 - zero installs, fully offline.
"""
import os
import subprocess
import time

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_TMP_DIR = os.path.join(PROJECT_DIR, ".ocr_tmp")

# PowerShell helper shipped with the windows-ocr-without-vision Hermes skill.
# Stored with forward slashes; os.path.normpath converts to backslashes at call
# time because the WinRT StorageFile API rejects forward-slash paths.
WINDOWS_OCR_PS1 = "C:/Users/Admin/AppData/Local/hermes/skills/windows/windows-ocr-without-vision/scripts/ocr_image.ps1"

_reader = None  # lazy easyocr.Reader singleton
_reader_error = None


# --------------------------------------------------------------------------- #
# easyocr
# --------------------------------------------------------------------------- #
def easyocr_models_ready():
    """True when both easyocr models are already cached (skips a ~100 MB download)."""
    root = os.path.expanduser(
        os.environ.get("EASYOCR_MODULE_PATH", os.path.join("~", ".EasyOCR"))
    )
    model_dir = os.path.join(root, "model")
    return os.path.exists(os.path.join(model_dir, "craft_mlt_25k.pth")) and os.path.exists(
        os.path.join(model_dir, "english_g2.pth")
    )


def get_reader():
    """Lazily build the easyocr Reader (CPU). Callers must hold the GIL-safe lock."""
    global _reader, _reader_error
    if _reader is not None:
        return _reader
    if _reader_error is not None:
        raise _reader_error
    try:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _reader
    except Exception as exc:  # download failure, torch error, ...
        _reader_error = exc
        raise


def _order_lines(results):
    """Sort easyocr boxes into reading order: rows by y, then left-to-right.

    results: list of (box[4pts], text, confidence).
    Returns (list_of_row_strings, avg_confidence, n_boxes).
    """
    if not results:
        return [], 0.0, 0
    items = []
    for box, text, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append((min(ys), min(xs), max(ys) - min(ys), box, text, conf))
    items.sort(key=lambda t: (t[0], t[1]))
    heights = [t[2] for t in items]
    mh = float(np.median(heights)) if heights else 20.0

    rows = []  # each row: [y_top, [items...]]
    for y, x, h, box, text, conf in items:
        if rows and y - rows[-1][0] <= max(10.0, mh * 0.6):
            rows[-1][1].append((x, text, conf))
        else:
            rows.append([y, [(x, text, conf)]])
    lines = []
    confs = []
    for _, row in rows:
        row.sort(key=lambda t: t[0])
        lines.append(" ".join(t[1] for t in row))
        confs.extend(t[2] for t in row)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return lines, avg_conf, len(results)


def _ocr_easyocr(img_rgb):
    """Run easyocr on an RGB (or 2D gray) numpy array."""
    import easyocr  # noqa: F401  (touch import so errors surface here)

    reader = get_reader()
    feed = img_rgb
    if feed.ndim == 3:
        feed = cv2.cvtColor(feed, cv2.COLOR_RGB2BGR)  # easyocr expects cv2/BGR arrays
    raw = reader.readtext(feed, detail=1, paragraph=False)
    lines, avg_conf, n = _order_lines(raw)
    boxes = [[[float(px), float(py)] for px, py in quad] for quad, _t, _c in raw]
    return lines, avg_conf, n, boxes


# --------------------------------------------------------------------------- #
# Windows built-in OCR (fallback)
# --------------------------------------------------------------------------- #
def _ocr_windows(img_rgb):
    """Windows.Media.Ocr via ocr_image.ps1. Requires a real file path (backslashes)."""
    if not os.path.exists(WINDOWS_OCR_PS1):
        raise FileNotFoundError("Windows OCR helper script not found: %s" % WINDOWS_OCR_PS1)
    os.makedirs(OCR_TMP_DIR, exist_ok=True)
    tmp_png = os.path.join(OCR_TMP_DIR, "winocr_%d.png" % int(time.time() * 1000))
    cv2.imwrite(tmp_png, img_rgb[:, :, ::-1] if img_rgb.ndim == 3 else img_rgb)

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        os.path.normpath(WINDOWS_OCR_PS1),  # backslash path (WinRT requirement)
        os.path.normpath(tmp_png),
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=180, **kwargs
        )
    finally:
        try:
            os.remove(tmp_png)
        except OSError:
            pass

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if "OCR_ERROR" in out or "NO_OCR_ENGINE" in out:
        raise RuntimeError(out if out else err)
    if "NO_TEXT_FOUND" in out or not out:
        return [], None, 0, None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines, None, len(lines), None  # no per-box confidence / geometry


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def ocr_image(img_rgb):
    """Extract text. Prefers easyocr, transparently falls back to Windows OCR.

    Returns dict: backend, lines, text, avg_conf, n_boxes, boxes, elapsed, note.
    """
    notes = []
    t0 = time.time()
    if easyocr_models_ready():
        try:
            lines, avg_conf, n, boxes = _ocr_easyocr(img_rgb)
            if lines:
                return _result(
                    "easyocr", lines, avg_conf, n, boxes, t0,
                    note="easyocr 1.7.2 (CPU) - models cached in ~/.EasyOCR/model",
                )
            notes.append("easyocr ran but found no text")
        except Exception as exc:
            notes.append("easyocr failed: %s" % exc)
    else:
        notes.append("easyocr models not cached yet - skipping (~100 MB download)")

    notes.append("falling back to Windows built-in OCR")
    try:
        lines, avg_conf, n, boxes = _ocr_windows(img_rgb)
        if lines:
            return _result(
                "windows", lines, avg_conf, n, boxes, t0,
                note="Windows.Media.Ocr (built-in, offline) - " + "; ".join(notes),
            )
        notes.append("Windows OCR found no text either")
    except Exception as exc:
        notes.append("Windows OCR failed: %s" % exc)

    return {
        "backend": None,
        "lines": [],
        "text": "",
        "avg_conf": None,
        "n_boxes": 0,
        "boxes": None,
        "elapsed": time.time() - t0,
        "note": "; ".join(notes),
        "error": "; ".join(notes),
    }


def _result(backend, lines, avg_conf, n, boxes, t0, note):
    return {
        "backend": backend,
        "lines": lines,
        "text": "\n".join(lines),
        "avg_conf": avg_conf,
        "n_boxes": n,
        "boxes": boxes,
        "elapsed": time.time() - t0,
        "note": note,
        "error": None,
    }


# --------------------------------------------------------------------------- #
# Preprocessing + region drawing
# --------------------------------------------------------------------------- #
def preprocess(img_rgb, grayscale=False, threshold=False, upscale=False, scale=2):
    """Apply optional preprocessing. Returns RGB/BGR numpy image ready for OCR.

    Order: upscale -> grayscale -> threshold (threshold forces grayscale).
    """
    out = img_rgb.copy()
    if upscale and scale > 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if (grayscale or threshold) and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    if threshold:
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
    return out


def draw_regions(img_rgb_or_gray, boxes):
    """Return a color RGB image with detected text regions outlined in green."""
    if img_rgb_or_gray.ndim == 2:
        base = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        base = img_rgb_or_gray.copy()
    for quad in boxes:
        pts = np.array(quad, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(base, [pts], isClosed=True, color=(0, 230, 118), thickness=2)
    return base
