# app/app.py
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from app.inference import predict_from_bytes



st.set_page_config(page_title="TraceFinder - Scanner ID", page_icon="🖨️")
st.title("TraceFinder: Forensic Scanner Identification")
st.write("Upload a scanned image to identify the source scanner and confidence score.")

uploaded = st.file_uploader("Upload image", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        with st.spinner("Running inference..."):
            result = predict_from_bytes(uploaded.read())
        st.success(f"Identified Scanner: {result['label']} | Confidence: {result['confidence']:.2f}%")
        with st.expander("Top-3 details"):
            for name, c in result["top3"]:
                st.write(f"- {name}: {c:.2f}%")
    except Exception as e:
        st.error(f"Inference failed: {e}")
