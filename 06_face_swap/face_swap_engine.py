"""
Face Swap Engine (no streamlit imports -> headless-testable).

Pipeline (primary):  mediapipe FaceLandmarker (tasks API, v1.x) -> ~130 face
landmark points on source & target -> shared Delaunay triangulation -> per-triangle
affine warp maps the SOURCE face texture onto the TARGET face geometry -> per-channel
color transfer (source skin tone matched to target lighting) -> Poisson seamless clone
of the face-hull region onto the target photo.

Fallback (OpenCV-only, if mediapipe missing / no mesh found): Haar frontal-face
boxes + eye detection -> partial-affine (similarity) warp source->target ->
elliptical mask -> seamless clone.

CPU only. Mediapipe model: mediapipe_models/face_landmarker.task (resolved next to
this file).
"""

import os
import time

import numpy as np
import cv2

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_ENGINE_DIR, "mediapipe_models", "face_landmarker.task")

MAX_DIM = 1100  # working resolution cap (keeps CPU warp fast, demo-grade output)

# ----------------------------------------------------------------------------
# mediapipe availability (imported lazily so the Haar fallback works without it)
# ----------------------------------------------------------------------------
def mediapipe_status():
    """Return dict: available(bool), model_exists(bool), version(str|None)."""
    info = {"available": False, "model_exists": os.path.isfile(_MODEL_PATH), "version": None}
    try:
        import mediapipe as mp  # noqa: F401
        info["available"] = True
        info["version"] = getattr(mp, "__version__", "?")
    except Exception:
        pass
    return info


_LANDMARKER = None


def _get_landmarker():
    """Lazily created FaceLandmarker singleton (mediapipe 1.x tasks API)."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    if not os.path.isfile(_MODEL_PATH):
        raise FileNotFoundError(f"FaceLandmarker model missing: {_MODEL_PATH}")
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision as mp_vision

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


# ----------------------------------------------------------------------------
# Image IO / prep
# ----------------------------------------------------------------------------
def decode_image(data):
    """Decode JPEG/PNG/... bytes -> BGR uint8 ndarray (alpha composited on white)."""
    if isinstance(data, np.ndarray):
        img = data
    else:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image - unsupported format?")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def _working_size(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def to_png_bytes(bgr_img):
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


# ----------------------------------------------------------------------------
# Landmark / face detection
# ----------------------------------------------------------------------------
def _mesh_points(bgr):
    """mediapipe FaceLandmarker -> (478,2) float32 pixel points or None."""
    try:
        landmarker = _get_landmarker()
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mimg)
        if not res.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]],
                       dtype=np.float32)
        return pts
    except Exception:
        return None


def _haar_faces(bgr):
    """Largest face box (x,y,w,h) found by Haar cascades, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    best = None
    for cname in ("haarcascade_frontalface_default.xml",
                  "haarcascade_frontalface_alt2.xml"):
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + cname)
        faces = cc.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            if best is None or w * h > best[2] * best[3]:
                best = (int(x), int(y), int(w), int(h))
    return best


def _haar_eyes(bgr, box):
    """Two largest eyes inside box -> ((lx,ly),(rx,ry)) centers or None."""
    x, y, w, h = box
    roi = bgr[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = cc.detectMultiScale(gray, 1.1, 4, minSize=(max(8, w // 10), max(8, w // 10)))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])  # left-to-right
    lc = (x + eyes[0][0] + eyes[0][2] / 2.0, y + eyes[0][1] + eyes[0][3] / 2.0)
    rc = (x + eyes[1][0] + eyes[1][2] / 2.0, y + eyes[1][1] + eyes[1][3] / 2.0)
    return (np.float32(lc), np.float32(rc))


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
# Face-landmark subset (face oval + brows + eyes + nose + lips), order kept.
# Standard mediapipe index set used for face warp / swap pipelines.
FACE_INDICES = list(dict.fromkeys([
    # outer face oval (forehead -> right jaw -> chin -> left jaw -> forehead)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
    # brows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # eyes (no iris)
    33, 7, 163, 144, 145, 153, 154, 155, 133, 362, 382, 381, 380, 374, 373,
    390, 249, 263,
    # nose
    1, 2, 98, 327, 49, 279, 97, 326, 168, 6, 195, 5, 4,
    # lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87,
    14, 317, 402, 318, 324, 308,
]))


def _mask_from_points(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.int32(cv2.convexHull(pts.astype(np.float32))), 255)
    return m


def _color_transfer(src, tgt, poly_src, poly_tgt):
    """Match source face tone/contrast to target face (per BGR channel)."""
    m1 = _mask_from_points(src.shape, poly_src)
    m2 = _mask_from_points(tgt.shape, poly_tgt)
    s1 = src[m1 > 0].reshape(-1, 3).astype(np.float32)
    s2 = tgt[m2 > 0].reshape(-1, 3).astype(np.float32)
    if len(s1) < 80 or len(s2) < 80:
        return src
    ms, mt = s1.mean(axis=0), s2.mean(axis=0)
    ss, st = s1.std(axis=0) + 1e-6, s2.std(axis=0) + 1e-6
    ratio = np.clip(st / ss, 0.5, 2.2)
    out = (src.astype(np.float32) - ms) * ratio + mt
    return np.clip(out, 0, 255).astype(np.uint8)


def _composite(canvas, tgt, mask255):
    """Poisson seamless clone (NORMAL_CLONE), alpha blend fallback if it fails."""
    try:
        ys, xs = np.nonzero(mask255)
        center = (int(xs.mean()), int(ys.mean()))
        return cv2.seamlessClone(canvas, tgt, mask255, center, cv2.NORMAL_CLONE)
    except Exception:
        m = cv2.GaussianBlur(mask255, (0, 0), 3).astype(np.float32) / 255.0
        return (tgt.astype(np.float32) * (1 - m[..., None]) +
                canvas.astype(np.float32) * m[..., None]).astype(np.uint8)


def _delaunay_tris(pts):
    """Subdiv2D Delaunay over pts (float32 Nx2). Returns list of index triples."""
    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    pad = 8
    rect = (max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad)
    subdiv = cv2.Subdiv2D(rect)
    for (px, py) in pts:
        subdiv.insert((float(px), float(py)))
    tris = []
    for t in subdiv.getTriangleList():
        tri = t.reshape(3, 2)
        idxs = []
        for v in tri:
            d = np.abs(pts - v).sum(axis=1)
            i = int(np.argmin(d))
            if d[i] > 1.5:  # vertex not in our point set -> skip triangle
                idxs = None
                break
            idxs.append(i)
        if idxs is None:
            continue
        a, b, c = pts[idxs]
        if abs(cv2.contourArea(np.int32([a, b, c]))) < 4:
            continue
        tris.append(tuple(idxs))
    return tris


def _warp_by_triangles(src_img, src_pts, tgt_pts, tris, tgt_shape):
    """Per-triangle affine warp of src_img so src_pts land on tgt_pts."""
    h, w = tgt_shape[:2]
    canvas = np.zeros((h, w, 3), np.uint8)
    for a, b, c in tris:
        st = np.float32([src_pts[a], src_pts[b], src_pts[c]])
        dt = np.float32([tgt_pts[a], tgt_pts[b], tgt_pts[c]])
        x0, y0 = np.floor(dt.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dt.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        M = cv2.getAffineTransform(st, dt)
        M[:, 2] -= [x0, y0]
        warped = cv2.warpAffine(src_img, M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mm, np.int32(dt - np.array([x0, y0], np.float32)), 255)
        roi = canvas[y0:y1, x0:x1]
        np.copyto(roi, warped, where=mm[:, :, None].astype(bool))
    return canvas


# ----------------------------------------------------------------------------
# Main swap (primary + fallback)
# ----------------------------------------------------------------------------
def swap_faces(src_data, tgt_data):
    """Swap SOURCE face onto TARGET photo. Returns (result_bgr, info dict)."""
    t0 = time.time()
    src = _working_size(decode_image(src_data))
    tgt = _working_size(decode_image(tgt_data))

    # -- try landmark path ---------------------------------------------------
    pts_src, pts_tgt = _mesh_points(src), _mesh_points(tgt)
    if pts_src is not None and pts_tgt is not None:
        result, method = _swap_landmarks(src, tgt, pts_src, pts_tgt)
    else:
        result, method = _swap_haar(src, tgt)

    info = {
        "method": method,
        "time_ms": int((time.time() - t0) * 1000),
        "src_shape": (src.shape[1], src.shape[0]),
        "tgt_shape": (tgt.shape[1], tgt.shape[0]),
        "out_shape": (result.shape[1], result.shape[0]),
    }
    return result, info


def _swap_landmarks(src, tgt, pts_src, pts_tgt):
    """Mediapipe-landmark + Delaunay warp face swap."""
    n = min(len(pts_src), len(pts_tgt))
    idx = [i for i in FACE_INDICES if i < n]
    S = pts_src[idx]  # face points in source
    T = pts_tgt[idx]  # face points in target

    # anchors: bounding-rect corners of each face point cloud (shared by index)
    bs = cv2.boundingRect(S.astype(np.float32))
    bt = cv2.boundingRect(T.astype(np.float32))
    corners_s = np.float32([[bs[0], bs[1]], [bs[0] + bs[2], bs[1]],
                            [bs[0] + bs[2], bs[1] + bs[3]], [bs[0], bs[1] + bs[3]]])
    corners_t = np.float32([[bt[0], bt[1]], [bt[0] + bt[2], bt[1]],
                            [bt[0] + bt[2], bt[1] + bt[3]], [bt[0], bt[1] + bt[3]]])
    S2 = np.vstack([S, corners_s])
    T2 = np.vstack([T, corners_t])

    # color-transfer corrected source texture
    src_c = _color_transfer(src, tgt, S, T)

    # Delaunay on target geometry (shared index triangles)
    tris = _delaunay_tris(T2)
    n_face = len(S)
    tris = [tr for tr in tris if sum(1 for i in tr if i < n_face) >= 2]

    canvas = _warp_by_triangles(src_c, S2, T2, tris, tgt.shape)

    mask = _mask_from_points(tgt.shape, T)
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    result = _composite(canvas, tgt, mask)
    return result, "mediapipe FaceLandmarker + Delaunay warp"


def _swap_haar(src, tgt):
    """OpenCV-only fallback: Haar boxes/eyes -> similarity warp -> ellipse blend."""
    bs, bt = _haar_faces(src), _haar_faces(tgt)
    if bs is None or bt is None:
        raise ValueError(
            "No face detected. Use a clear, front-facing photo of one person "
            "(good lighting, face filling most of the frame)."
        )
    es, et = _haar_eyes(src, bs), _haar_eyes(tgt, bt)

    def box_pts(box, eye):
        x, y, w, h = box
        pts = []
        if eye:
            pts += [eye[0], eye[1], ((eye[0][0] + eye[1][0]) / 2, (eye[0][1] + eye[1][1]) / 2)]
        pts.append((x + w / 2.0, y + h * 0.62))  # chin estimate
        return np.float32(pts)

    ps, pt = box_pts(bs, es), box_pts(bt, et)
    if len(ps) == len(pt) and len(ps) >= 2:
        M, _ = cv2.estimateAffinePartial2D(ps, pt)
    else:  # geometry from box centers only (no eyes)
        cs = (bs[0] + bs[2] / 2.0, bs[1] + bs[3] / 2.0)
        ct = (bt[0] + bt[2] / 2.0, bt[1] + bt[3] / 2.0)
        s = (bt[2] / float(bs[2]) + bt[3] / float(bs[3])) / 2.0
        M = np.float32([[s, 0, ct[0] - s * cs[0]], [0, s, ct[1] - s * cs[1]]])
    if M is None:
        raise ValueError("Could not align the two faces.")

    # color-correct source using box regions, then warp full frame
    src_c = _color_transfer(src, tgt,
                            np.float32([[bs[0], bs[1]], [bs[0] + bs[2], bs[1]],
                                        [bs[0] + bs[2], bs[1] + bs[3]], [bs[0], bs[1] + bs[3]]]),
                            np.float32([[bt[0], bt[1]], [bt[0] + bt[2], bt[1]],
                                        [bt[0] + bt[2], bt[1] + bt[3]], [bt[0], bt[1] + bt[3]]]))
    h, w = tgt.shape[:2]
    canvas = cv2.warpAffine(src_c, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    x, y, bw, bh = bt
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(x + bw / 2), int(y + bh * 0.44)),
                (int(bw * 0.46), int(bh * 0.55)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    result = _composite(canvas, tgt, mask)
    return result, "OpenCV Haar (fallback)"
