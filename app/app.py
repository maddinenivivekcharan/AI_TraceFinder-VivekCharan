# app/app.py

import os, re, glob, math, json, pickle
from pathlib import Path
import numpy as np
import streamlit as st
import cv2, pywt
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

# ---------------- CONFIG ----------------
APP_TITLE = "TraceFinder - Forensic Scanner Identification"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16  # used for features and aggregation

# ---------------- Paths (repo-relative) ----------------
BASE_DIR = Path(__file__).resolve().parent
ART_SCN = BASE_DIR / "models"                          # scanner artifacts
ART_IMG = ART_SCN                                      # image-level tamper artifacts live directly under models
ART_PAIR = ART_SCN / "artifacts_tamper_pair"           # paired tamper
TAMP_ROOT = ART_SCN / "Tampered images"                # originals for pairing

def must_exist(p: Path, kind="file"):
    if kind == "file" and not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p}")
    if kind == "dir" and not p.is_dir():
        raise FileNotFoundError(f"Missing required folder: {p}")
    return p

# ---------------- PDF via PyMuPDF only ----------------
PDF_BACKEND = "pymupdf"
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False
    PDF_BACKEND = None

def pdf_bytes_to_bgr(file_bytes: bytes):
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PDF support not available. Add 'pymupdf' to requirements.txt and redeploy.")
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ---------------- Page ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<h2 style='margin-top:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)

# ---------------- Image utils ----------------
def decode_upload_to_bgr(uploaded):
    try:
        uploaded.seek(0)
    except Exception:
        pass
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        bgr = pdf_bytes_to_bgr(raw)
        return bgr, name
    buf = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode file")
    return bgr, name

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (gray - den).astype(np.float32)

def extract_patches(res, patch=PATCH, stride=STRIDE, limit=MAX_PATCHES, seed=42):
    H, W = res.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    coords = [(y, x) for y in ys for x in xs]
    rng = np.random.RandomState(seed)
    rng.shuffle(coords)
    coords = coords[:min(limit, len(coords))]
    return [res[y:y + patch, x:x + patch] for y, x in coords]

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, K + 1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i + 1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return np.asarray(feats, dtype=np.float32)

# ---------------- Scanner-ID (14-class artifacts) ----------------
hyb_model = None
scanner_ready = False
scanner_err = None
try:
    import tensorflow as tf  # requires tensorflow-cpu in requirements
    cand = [ART_SCN / "scanner_hybrid_14.keras",
            ART_SCN / "scanner_hybrid.keras",   # fallback if 14 not present
            ART_SCN / "scanner_hybrid.h5",
            ART_SCN / "scanner_hybrid"]
    found = next((p for p in cand if p.exists()), None)
    if found:
        hyb_model = tf.keras.models.load_model(str(found))
    else:
        scanner_err = "No scanner_hybrid model file found under app/models."
except Exception as e:
    scanner_err = f"TF model load failed: {e}"

try:
    # Prefer 14-class names; fall back to legacy if needed
    LE_CAND = [ART_SCN / "hybrid_label_encoder.pkl", ART_SCN / "hybrid_label_encoder (1).pkl"]
    SC_PATH = must_exist(ART_SCN / "hybrid_feat_scaler.pkl")
    FPS_CAND = [ART_SCN / "scanner_fingerprints_14.pkl", ART_SCN / "scanner_fingerprints.pkl"]
    FPK_CAND = [ART_SCN / "fp_keys_14.npy", ART_SCN / "fp_keys.npy"]

    LE_PATH = next((p for p in LE_CAND if p.exists()), None)
    FPS_PATH = next((p for p in FPS_CAND if p.exists()), None)
    FPK_PATH = next((p for p in FPK_CAND if p.exists()), None)
    if not LE_PATH or not FPS_PATH or not FPK_PATH:
        raise FileNotFoundError("Required scanner artifacts not found (label encoder / fingerprints / fp_keys).")

    with open(LE_PATH, "rb") as f: le_sc = pickle.load(f)
    with open(SC_PATH, "rb") as f: sc_sc = pickle.load(f)
    with open(FPS_PATH, "rb") as f: fps = pickle.load(f)
    fp_keys = np.load(FPK_PATH, allow_pickle=True).tolist()
    if not isinstance(fp_keys, (list, tuple)) or len(fp_keys) == 0:
        raise ValueError("fp_keys empty or invalid")
    if any(k not in fps for k in fp_keys):
        missing = [k for k in fp_keys if k not in fps]
        raise ValueError(f"Missing fingerprints for keys: {missing[:3]} ...")
    scanner_ready = (hyb_model is not None)
except Exception as e:
    scanner_err = f"Scanner artifacts problem: {e}"
    scanner_ready = False

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def make_scanner_feats(res):
    v_corr = [corr2d(res, fps[k]) for k in fp_keys]  # length matches the selected fp_keys file (14-class if available)
    v_fft  = fft_radial_energy(res, K=6).tolist()
    v_lbp  = lbp_hist_safe(res, P=8, R=1.0).tolist()
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    return sc_sc.transform(v)

def try_scanner_predict(residual):
    if not scanner_ready:
        if scanner_err:
            st.info(f"Scanner-ID disabled: {scanner_err}")
        return "Unknown", 0.0
    try:
        x_img = np.expand_dims(residual, axis=(0, -1))
        x_ft  = make_scanner_feats(residual)
        preds = hyb_model.predict([x_img, x_ft], verbose=0)
        ps = preds.ravel()
        s_idx = int(np.argmax(ps))
        return str(le_sc.classes_[s_idx]), float(ps[s_idx] * 100.0)
    except Exception as e:
        st.warning(f"Scanner-ID inference error: {e}")
        return "Unknown", 0.0

# ---------------- Image-level tamper (preferred artifacts) ----------------
tamper_image_ok = True
try:
    # You have these three files in models per your screenshot
    with must_exist(ART_IMG / "image_scaler.pkl").open("rb") as f: sc_img = pickle.load(f)
    with must_exist(ART_IMG / "image_svm_sig.pkl").open("rb") as f: clf_img = pickle.load(f)
    # Some systems may name it "image_thresholds.json" or "image_thresholds"
    THR_PATH = ART_IMG / "image_thresholds.json"
    if not THR_PATH.exists():
        THR_PATH = ART_IMG / "image_thresholds"
    with THR_PATH.open("r") as f: THR_IMG = json.load(f)
except Exception as e:
    tamper_image_ok = False
    st.info(f"Tamper (image-level) disabled: {e}")

def image_feat_mean(res):
    patches = extract_patches(res, limit=MAX_PATCHES, seed=111)
    feats = []
    for p in patches:
        lbp = lbp_hist_safe(p, 8, 1.0)
        fft6 = fft_radial_energy(p, 6)
        c1 = float(np.std(p)); c2 = float(np.mean(np.abs(p - np.mean(p))))
        feats.append(np.concatenate([lbp, fft6, np.array([c1, c2], np.float32)], 0))
    if len(feats) == 0:
        feats = [np.zeros(18, np.float32)]
    m = np.mean(np.stack(feats, 0), axis=0).reshape(1, -1)
    return m

def choose_thr_image(domain):
    if not tamper_image_ok:
        return 0.5
    thr = THR_IMG.get("global", 0.5)
    byd = THR_IMG.get("by_domain", {})
    return byd.get(domain, thr)

def infer_tamper_image_from_residual(residual, domain):
    if not tamper_image_ok:
        return 0, 0.0, 0.5
    x = sc_img.transform(image_feat_mean(residual))
    p = float(clf_img.predict_proba(x)[0, 1])
    thr = choose_thr_image(domain)
    tampered = int(p >= thr)
    return tampered, p, thr

# ---------------- Paired tamper (optional) ----------------
tamper_pair_ok = True
try:
    with must_exist(ART_PAIR / "pair_scaler.pkl").open("rb") as f: sc_pair = pickle.load(f)
    with must_exist(ART_PAIR / "pair_svm_sig.pkl").open("rb") as f: pair_clf = pickle.load(f)
    with must_exist(ART_PAIR / "pair_thresholds_topk.json").open("r") as f: THR_PAIR = json.load(f)
except Exception as e:
    tamper_pair_ok = False
    st.info(f"Tamper-pair disabled: {e}")

def pid_from_name(p):
    m = re.search(r"(s\d+_\d+)", os.path.basename(p))
    return m.group(1) if m else None

def build_orig_index():
    try:
        orig_glob = glob.glob(str(TAMP_ROOT / "Original" / "*.tif"))
        return {pid_from_name(p): p for p in orig_glob if pid_from_name(p)}
    except Exception:
        return {}

orig_map = build_orig_index()

# ---------------- Domain/type inference ----------------
def infer_domain_and_type_from_path_or_name(path_or_name: str):
    p = path_or_name.replace("\\", "/").lower()
    if "/adobescan/" in p or "/geniusscan/" in p or "/tinyscanner/" in p:
        return "mobile_scan", None
    if "/tampered images/original/" in p: return "orig_pdf_tif", None
    if "/originals_tif/official/" in p:    return "orig_pdf_tif", None
    if "/originals_tif/wikipedia/" in p:   return "orig_pdf_tif", None
    if "/tampered images/tampered/copy-move/" in p: return "tamper_dir", "copy-move"
    if "/tampered images/tampered/retouching/" in p: return "tamper_dir", "retouch"
    if "/tampered images/tampered/splicing/" in p:   return "tamper_dir", "splice"
    if re.search(r"_a(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p): return "tamper_dir", "splice"
    if re.search(r"_b(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p): return "tamper_dir", "copy-move"
    if re.search(r"_c(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p): return "tamper_dir", "retouch"
    return "orig_pdf_tif", None

def paired_infer_type_aware(clean_path, suspect_residual, typ_hint):
    if not tamper_pair_ok or not clean_path:
        return 0, 0.0, 0.5, 0
    clean_bgr = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
    r1 = load_to_residual_from_bgr(clean_bgr)
    patches1 = extract_patches(r1, limit=MAX_PATCHES, seed=777)
    patches2 = extract_patches(suspect_residual, limit=MAX_PATCHES, seed=777)
    n = min(len(patches1), len(patches2))
    Xd = []
    def residual_stats(img):
        return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)
    def fft_resample_feats(img):
        f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        b1 = (r>=0.25*rmax) & (r<0.35*rmax)
        b2 = (r>=0.35*rmax) & (r<0.50*rmax)
        e1 = float(mag[b1].mean() if b1.any() else 0.0)
        e2 = float(mag[b2].mean() if b2.any() else 0.0)
        ratio = float(e2/(e1+1e-8))
        return np.asarray([e1,e2,ratio], dtype=np.float32)
    for i in range(n):
        f1 = np.concatenate([lbp_hist_safe(patches1[i],8,1.0), fft_radial_energy(patches1[i],6), residual_stats(patches1[i]), fft_resample_feats(patches1[i])], 0)
        f2 = np.concatenate([lbp_hist_safe(patches2[i],8,1.0), fft_radial_energy(patches2[i],6), residual_stats(patches2[i]), fft_resample_feats(patches2[i])], 0)
        Xd.append(f2 - f1)
    Xd = np.asarray(Xd, np.float32)
    Xd_s = sc_pair.transform(Xd)
    p_patch = pair_clf.predict_proba(Xd_s)[:, 1]
    typ = (typ_hint or "unknown").lower()
    thr_base = THR_PAIR.get("by_type", {}).get(typ, THR_PAIR.get("global", 0.5))
    n = len(p_patch); k = max(1, int(math.ceil(0.30 * n)))
    top_idx = np.argsort(p_patch)[-k:]
    p_img = float(np.mean(p_patch[top_idx]))
    hits = int(np.sum(p_patch[top_idx] >= 0.80))
    ok = (p_img >= thr_base) and (hits >= 2)
    return int(ok), p_img, thr_base, hits

# ---------------- UI ----------------
st.write("")
uploaded = st.file_uploader(
    "Upload scanned page",
    type=["tif", "tiff", "png", "jpg", "jpeg", "pdf"],
    label_visibility="collapsed"
)

def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        st.image(rgb, width="stretch")   # deprecation fix
    except TypeError:
        st.image(rgb)

if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner ID (14-class if files present)
        s_lab, s_conf = try_scanner_predict(residual)

        # Tamper pipeline: prefer paired if original exists; otherwise image-level
        domain, typ_hint = infer_domain_and_type_from_path_or_name(display_name)
        pid = re.search(r"(s\d+_\d+)", display_name)
        pid = pid.group(1) if pid else None

        if pid and (pid in (orig_map or {})):
            domain = "orig_pdf_tif"
            typ_hint = None
            is_t, p_img, thr_used, hits = paired_infer_type_aware(orig_map[pid], residual, typ_hint)
        else:
            is_t, p_img, thr_used = infer_tamper_image_from_residual(residual, domain)
            hits = 0

        verdict = "Tampered" if is_t else "Clean"

        colL, colR = st.columns([1.2, 1.8], gap="large")
        with colR:
            safe_show_image(bgr)
        with colL:
            st.markdown(
                f"""
                <div style='padding:16px;border-radius:8px;background:#111317;border:1px solid #2a2f3a;'>
                    <div style='font-size:16px;color:#9aa4b2;'>Scanner</div>
                    <div style='font-size:20px;margin-top:4px;'>{s_lab}</div>
                    <div style='font-size:13px;color:#9aa4b2;margin-top:2px;'>{s_conf:.1f}% confidence</div>
                    <hr style='border:none;border-top:1px solid #2a2f3a;margin:12px 0;'>
                    <div style='font-size:16px;color:#9aa4b2;'>Tamper verdict</div>
                    <div style='font-size:20px;margin-top:4px;'>{verdict}</div>
                    <div style='font-size:12px;color:#9aa4b2;margin-top:8px;'>p={p_img:.3f} · thr={thr_used:.3f} · domain={domain} · hits={hits}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    except ImportError as e:
        st.error(str(e))
    except Exception as e:
        import traceback
        st.error("Inference error")
        st.code(traceback.format_exc())
else:
    st.info("Drag-and-drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
