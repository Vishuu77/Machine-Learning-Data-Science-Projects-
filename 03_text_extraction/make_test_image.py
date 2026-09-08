"""Generate a sample image with clear text for OCR testing."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 340
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# Windows ships arial.ttf; fall back to PIL default if unavailable
try:
    font_big = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 72)
    font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 44)
except Exception:
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()

lines = [
    ("EasyOCR Test 123", font_big),
    ("Streamlit OCR App - 2026", font_small),
]
y = 30
for text, font in lines:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) / 2 - bbox[0], y), text, fill="black", font=font)
    y += th + 40

out = "C:/Users/Admin/course_projects/03_text_extraction/sample_text.png"
img.save(out)
print("saved", out, img.size)
