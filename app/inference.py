import io, pickle
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from skimage.feature import local_binary_pattern as sk_lbp

from app.utils.preprocess import preprocess_residual_from_array

# Artifact paths
MODEL_PATH  = "app/models/scanner_hybrid.keras"
LE_PATH     = "app/models/hybrid_label_encoder.pkl"
SCALER_PATH = "app/models/hybrid_feat_scaler.pkl"
FPS_PATH    = "app/models/scanner_fingerprints.pkl"
FP_KEYS     = "app/models/fp_keys.npy"

# Load artifacts once
hyb_model  = tf.keras.models.load_model(MODEL_PATH)
le_inf     = joblib.load(LE_PATH)
scaler_inf = joblib.load(SCALER_PATH)
with open(FPS_PATH, "rb") as f:
    scanner_fps_inf = pickle.load(f)
fp_keys_inf = np.load(FP_KEYS, allow_pickle=True).tolist()

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    # NumPy 2.0: use np.ptp(img)
    rng = float(np.ptp(img))
    if rng < 1e-12:
        g = np.zeros_like(img, dtype=np.float32)
    else:
        g = (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype(np.float32).tolist()

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]  # 11
    v_fft  = fft_radial_energy(res, K=6)                              # 6
    v_lbp  = lbp_hist_safe(res, P=8, R=1.0)                           # 10
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    v = scaler_inf.transform(v)                                       # (1,27)
    return v

def _topk(prob, k=3):
    k = min(k, prob.size)
    idx = np.argpartition(prob, -k)[-k:]
    return idx[np.argsort(prob[idx])[::-1]]

def predict_from_bytes(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_np = np.array(img)

    res   = preprocess_residual_from_array(img_np)
    x_img = np.expand_dims(res, axis=(0, -1))
    x_ft  = make_feats_from_res(res)

    prob  = hyb_model.predict([x_img, x_ft], verbose=0)
    prob  = np.asarray(prob).ravel()

    idx   = int(np.argmax(prob))
    label = le_inf.classes_[idx]
    conf  = float(prob[idx] * 100.0)

    top3i = _topk(prob, 3)
    top3  = [(le_inf.classes_[i], float(prob[i]*100.0)) for i in top3i]
    return {"label": label, "confidence": conf, "top3": top3}
