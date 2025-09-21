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
MAX_PATCHES = 16

TOPK = 0.30
HIT_THR = 0.85
MIN_HITS = 2

# Repo-relative paths
BASE_DIR = Path(__file__).resolve().parent
ART_SCN = BASE_DIR / "models"
TAMP_ROOT = ART_SCN / "Tampered images"
ART_TP = ART_SCN / "artifacts_tamper_patch"
ART_PAIR = ART_SCN / "artifacts_tamper_pair"

def must_exist(p: Path, kind="file"):
    if kind == "file" and not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p}")
    if kind == "dir" and not p.is_dir():
        raise FileNotFoundError(f"Missing required folder: {p}")
    return p

# ---------------- PDF backend (PyMuPDF only) ----------------
PDF_BACKEND = "pymupdf"
try:
    import fitz  # PyMuPDF
except Exception:
    PDF_BACKEND = None

def pdf_bytes_to_bgr(file_bytes: bytes):
    if PDF_BACKEND != "pymupdf":
        raise ImportError("PDF support not available. Please add 'pymupdf' to requirements.txt.")
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

# ---------------- Low-level utils ----------------
def decode_upload_to_bgr(uploaded):
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        bgr = pdf_bytes_to_bgr(raw)
        return bgr, name
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
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

def residual_stats(img):
    return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    b1 = (r >= 0.25 * rmax) & (r < 0.35 * rmax)
    b2 = (r >= 0.35 * rmax) & (r < 0.50 * rmax)
    e1 = float(mag[b1].mean() if b1.any() else 0.0)
    e2 = float(mag[b2].mean() if b2.any() else 0.0)
    ratio = float(e2 / (e1 + 1e-8))
    return np.asarray([e1, e2, ratio], dtype=np.float32)

def make_feat_vector(img_patch):
    lbp = lbp_hist_safe(img_patch, 8, 1.0)
    fft6 = fft_radial_energy(img_patch, 6)
    res3 = residual_stats(img_patch)
    rsp3 = fft_resample_feats(img_patch)
    return np.concatenate([lbp, fft6, res3, rsp3], axis=0)

# ---------------- Domain/type inference (defined before use) ----------------
def infer_domain_and_type_from_path_or_name(path_or_name: str):
    p = path_or_name.replace("\\", "/").lower()
    # Original sources
    if "/tampered images/original/" in p:
        return "orig_pdf_tif", None
    if "/originals_tif/official/" in p:
        return "orig_pdf_tif", None
    if "/originals_tif/wikipedia/" in p:
        return "orig_pdf_tif", None
    # Tampered subfolders
    if "/tampered images/tampered/copy-move/" in p:
        return "tamper_dir", "copy-move"
    if "/tampered images/tampered/retouching/" in p:
        return "tamper_dir", "retouch"
    if "/tampered images/tampered/splicing/" in p:
        return "tamper_dir", "splice"
    # Suffix hints
    if re.search(r"_a(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p):
        return "tamper_dir", "splice"
    if re.search(r"_b(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p):
        return "tamper_dir", "copy-move"
    if re.search(r"_c(\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.pdf)$", p):
        return "tamper_dir", "retouch"
    # Default
    return "orig_pdf_tif", None

# ---------------- Scanner-ID ----------------
hyb_model = None
try:
    import tensorflow as tf
    cand = [ART_SCN / "scanner_hybrid.keras",
            ART_SCN / "scanner_hybrid.h5",
            ART_SCN / "scanner_hybrid"]
    found = next((p for p in cand if p.exists()), None)
    if found:
        hyb_model = tf.keras.models.load_model(str(found))
except Exception:
    hyb_model = None

# Required artifacts
LE_PATH   = ART_SCN / "hybrid_label_encoder.pkl"
SC_PATH   = ART_SCN / "hybrid_feat_scaler.pkl"
FPS_PATH  = ART_SCN / "scanner_fingerprints.pkl"
FPK_PATH  = ART_SCN / "fp_keys.npy"

try:
    with must_exist(LE_PATH).open("rb") as f: le_sc = pickle.load(f)
    with must_exist(SC_PATH).open("rb") as f: sc_sc = pickle.load(f)
    with must_exist(FPS_PATH).open("rb") as f: fps = pickle.load(f)
    fp_keys = np.load(must_exist(FPK_PATH), allow_pickle=True).tolist()
except Exception as e:
    st.error(f"Model artifacts missing under app/models. {e}")
    st.stop()

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def make_scanner_feats(res):
    v_corr = [corr2d(res, fps[k]) for k in fp_keys]
    v_fft  = fft_radial_energy(res, K=6)
    v_lbp  = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr + v_fft.tolist() + v_lbp.tolist(), dtype=np.float32).reshape(1, -1)
    return sc_sc.transform(v)

# ---------------- Single-image tamper ----------------
try:
    with must_exist(ART_TP / "patch_scaler.pkl").open("rb") as f: sc_tp = pickle.load(f)
    with must_exist(ART_TP / "patch_svm_sig_calibrated.pkl").open("rb") as f: clf_tp = pickle.load(f)
    with must_exist(ART_TP / "thresholds_patch.json").open("r") as f: THRS_TP = json.load(f)
except Exception as e:
    st.error(f"Tamper (single) artifacts missing: {e}")
    st.stop()

def choose_thr_single(domain, typ):
    if "by_type" in THRS_TP and typ in THRS_TP["by_type"]:
        return THRS_TP["by_type"][typ]
    if "by_domain" in THRS_TP and domain in THRS_TP["by_domain"]:
        return THRS_TP["by_domain"][domain]
    return THRS_TP.get("global", 0.5)

def image_score_topk(patch_probs, frac=TOPK):
    n = len(patch_probs); k = max(1, int(math.ceil(frac * n)))
    top = np.sort(np.asarray(patch_probs))[-k:]
    return float(np.mean(top))

def infer_tamper_single_from_residual(residual, domain, typ_hint):
    patches = extract_patches(residual, limit=MAX_PATCHES, seed=123)
    feats = np.stack([make_feat_vector(p) for p in patches], 0)
    feats = sc_tp.transform(feats)
    p_patch = clf_tp.predict_proba(feats)[:, 1]
    p_img = image_score_topk(p_patch, frac=TOPK)
    thr = choose_thr_single(domain, (typ_hint or "unknown"))
    if domain == "orig_pdf_tif":
        thr = min(1.0, thr + 0.03)
    hits = int((p_patch >= HIT_THR).sum())
    tampered = int((p_img >= thr) and (hits >= MIN_HITS))
    if domain == "orig_pdf_tif":
        tampered = 0
    return tampered, p_img, thr, hits

# ---------------- Paired tamper ----------------
try:
    with must_exist(ART_PAIR / "pair_scaler.pkl").open("rb") as f: sc_pair = pickle.load(f)
    with must_exist(ART_PAIR / "pair_svm_sig.pkl").open("rb") as f: pair_clf = pickle.load(f)
    with must_exist(ART_PAIR / "pair_thresholds_topk.json").open("r") as f: THR_PAIR = json.load(f)
except Exception as e:
    st.error(f"Tamper (pair) artifacts missing: {e}")
    st.stop()

def pid_from_name(p):
    m = re.search(r"(s\d+_\d+)", os.path.basename(p))
    return m.group(1) if m else None

def build_orig_index():
    orig_glob = glob.glob(str(TAMP_ROOT / "Original" / "*.tif"))
    return {pid_from_name(p): p for p in orig_glob if pid_from_name(p)}

orig_map = build_orig_index()

def paired_infer_type_aware(clean_path, suspect_residual, typ_hint):
    clean_bgr = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
    r1 = load_to_residual_from_bgr(clean_bgr)
    patches1 = extract_patches(r1, limit=MAX_PATCHES, seed=777)
    patches2 = extract_patches(suspect_residual, limit=MAX_PATCHES, seed=777)
    n = min(len(patches1), len(patches2))
    Xd = []
    for i in range(n):
        f1 = make_feat_vector(patches1[i])
        f2 = make_feat_vector(patches2[i])
        Xd.append(f2 - f1)
    Xd = np.asarray(Xd, np.float32)
    Xd_s = sc_pair.transform(Xd)
    p_patch = pair_clf.predict_proba(Xd_s)[:, 1]
    typ = (typ_hint or "unknown").lower()
    thr_base = THR_PAIR.get("by_type", {}).get(typ, THR_PAIR.get("global", 0.5))
    thr_eff = max(thr_base - 0.02, 0.0)
    frac_use = 0.20 if typ == "retouch" else 0.30
    n = len(p_patch); k = max(1, int(math.ceil(frac_use * n)))
    top_idx = np.argsort(p_patch)[-k:]
    p_img = float(np.mean(p_patch[top_idx]))
    def hits_topk(gate): return int(np.sum(p_patch[top_idx] >= gate))
    if typ == "copy-move":
        local_gate = 0.78
        hits = hits_topk(local_gate)
        ok = (p_img >= (thr_eff - 0.05)) or ((p_img >= thr_eff) and (hits >= 2))
        thr_used = thr_eff - 0.05
    elif typ == "retouch":
        local_gate = 0.75
        hits = hits_topk(local_gate)
        ok = (p_img >= 0.60) or ((p_img >= max(thr_eff, 0.65)) and (hits >= 2))
        thr_used = 0.60
    else:
        local_gate = 0.80
        hits = hits_topk(local_gate)
        ok = (p_img >= thr_base) and (hits >= 2)
        thr_used = thr_base
    return int(ok), p_img, thr_used, hits

# ---------------- UI ----------------
st.write("")
uploaded = st.file_uploader(
    "Upload scanned page",
    type=["tif", "tiff", "png", "jpg", "jpeg", "pdf"],
    label_visibility="collapsed"
)

if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner ID
        s_lab, s_conf = "Unknown", 0.0
        try:
            if 'tf' in globals() and hyb_model is not None:
                x_img = np.expand_dims(residual, axis=(0, -1))
                x_ft  = make_scanner_feats(residual)
                ps = hyb_model.predict([x_img, x_ft], verbose=0).ravel()
                s_idx = int(np.argmax(ps)); s_lab = le_sc.classes_[s_idx]; s_conf = float(ps[s_idx] * 100.0)
        except Exception:
            pass

        # Domain/type inference
        domain, typ_hint = infer_domain_and_type_from_path_or_name(display_name)

        # Prefer paired if Original reference exists for this PID
        pid = pid_from_name(display_name)
        if pid and (pid in orig_map):
            domain = "orig_pdf_tif"
            typ_hint = None
            is_t, p_img, thr_used, hits = paired_infer_type_aware(orig_map[pid], residual, typ_hint)
        else:
            is_t, p_img, thr_used, hits = infer_tamper_single_from_residual(residual, domain, typ_hint)

        verdict = "Tampered" if is_t else "Clean"

        colL, colR = st.columns([1.2, 1.8], gap="large")
        with colR:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
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
