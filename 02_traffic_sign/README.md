# 🚦 German Traffic Sign Classifier

Streamlit + PyTorch (CPU-only) mini project: upload a photo of a German traffic sign
and get the **top-3 predictions with confidence bars** and the **meaning of the sign**
(German + English), powered by a small CNN trained from scratch on a public GTSRB subset.

## Files

| file            | purpose                                                              |
|-----------------|----------------------------------------------------------------------|
| `train_model.py`| downloads the dataset, trains the CNN, exports `model.pt` + `meta.json` + `samples/` |
| `app.py`        | Streamlit app: upload → top-3 predictions, confidence chart, meanings |
| `sign_info.py`  | GTSRB class id → English name / German name / meaning (43 classes)    |
| `.streamlit/config.toml` | dark theme configuration                                     |
| `model.pt`      | trained TorchScript CNN (softmax probabilities out)                   |
| `meta.json`     | classes, labels, meanings, validation accuracy, dataset info          |
| `samples/`      | 4 demo images (cropped from the validation split)                     |

## Quick start

```bash
cd course_projects/02_traffic_sign
python train_model.py                     # download + train (CPU, a few minutes)
python -m streamlit run app.py --server.port 2002
```

Open http://localhost:2002 — a sample image is classified automatically;
upload your own sign photo or pick another bundled sample.

## Dataset

[tanganke/gtsrb](https://huggingface.co/datasets/tanganke/gtsrb) on Hugging Face —
the classic **GTSRB** (German Traffic Sign Recognition Benchmark, Stallkamp et al.,
IJCNN 2011) re-hosted as images + labels, **no authentication required**.
Label ids equal the official GTSRB classes (0–42).

Training uses a balanced 12-class subset (iconic German signs: speed limits 30/50/60,
priority road, yield, stop, no entry, road work, children crossing, wild animals,
ahead only, roundabout), 190 train + 50 validation images per class,
a small 3-conv-layer CNN at 48×48 RGB, ~10 epochs on CPU.
