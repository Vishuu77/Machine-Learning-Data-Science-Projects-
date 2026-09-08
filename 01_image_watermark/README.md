# 🖼️ Watermark Studio (Streamlit)

Non-destructive image watermarking: upload an image, add a **text watermark**
(tile or single placement, font size, opacity, rotation, colour, position) or a
**logo watermark** (scale, position, opacity), preview original vs watermarked
side-by-side, then download a lossless **PNG**.

The original upload is never modified — everything is composited on a copy.

## Run

```bash
cd C:/Users/Admin/course_projects/01_image_watermark
python -m streamlit run app.py --server.port 2001
```

Open http://localhost:2001

## Files

| File                 | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `app.py`             | Streamlit UI (dark theme, sidebar controls, preview, download)   |
| `watermark_engine.py`| Pure Pillow processing: text/logo watermarking, PNG encoding      |

## Notes / limitations

- Transparent PNG uploads are handled safely (RGBA pipeline); alpha is
  preserved in the output when the source had it, otherwise flattened to RGB.
- Animated GIFs use their first frame.
- Very small logos are up-scaled (LANCZOS); for crisp results upload a logo
  ≥ the size you want on screen.
- Text is normalised to a single line; use `Rotation (°)` for diagonal marks.
