TraceFinder — Scanner Identification & Tampering Detection

TraceFinder is an AI-powered forensic analysis project designed to identify the source scanner device used to scan a document image and detect tampering within scanned documents.
By analyzing unique intrinsic patterns, noise artifacts, texture, and frequency signals left by scanners, the system supports forensic investigation, document authentication, and legal verification workflows.

Objectives

Accurately identify scanner brand/model from residual image artifacts and handcrafted features using a hybrid CNN–ML approach.

Detect manipulations such as copy-move, retouching, and splicing tampering in scanned images using patch-level and paired-comparison ML classifiers.

Provide confidence scores on predictions to enable forensic reliability and transparency.

Use Cases

Digital Forensics: Attribute forged or duplicated document scans to their source scanner device.

Document Authentication: Distinguish authorized scanner outputs from unauthorized or manipulated ones.

Legal Evidence Verification: Confirm scanned copies used in legal contexts originate from approved devices, ensuring evidence integrity.

System Overview
Residual Preprocessing

Convert images to grayscale, resize to fixed resolution.

Denoise using Haar Discrete Wavelet Transform (DWT) by zeroing detail coefficients.

Compute residuals to enhance scanner/tamper signals.

Scanner Identification

Hybrid model combining a CNN on residual images and a 27-dimensional handcrafted feature vector.

Features include:

11 correlations to reference scanner fingerprints

6 frequency-domain FFT radial energies

10 Local Binary Pattern (LBP) texture histogram features

Model outputs scanner class with confidence scores.

Tampering Detection

Image-level classifier: Aggregates 18-dimensional patch descriptors (LBP + FFT + contrast stats) using a calibrated SVM.

Patch-level fallback: 22-dimensional patch features including residual and FFT resampling stats, with top-k patch probabilities for final decision.

Paired-image comparison: Uses differential patch features between suspect and original images for higher reliability.

Milestones and Timeline
Milestone 1 — Dataset Collection & Preprocessing (Weeks 1–2)

Collect and label scanned samples from multiple scanner models.

Normalize dataset structure and manifests.

Preprocess images with grayscale conversion, resizing, and residual computation.

Output labeled datasets and verified residual images.

Milestone 2 — Feature Engineering & Baseline Modeling (Weeks 3–4)

Extract handcrafted noise, frequency, and texture features.

Train baseline ML classifiers (SVM, Random Forest).

Evaluate performance via accuracy metrics and confusion matrices.

Generate visualization of scanner-specific residuals.

Milestone 3 — Deep Learning Model & Explainability (Weeks 5–6)

Implement hybrid CNN with residual image and handcrafted features as inputs.

Apply augmentation for better generalization.

Evaluate using accuracy, F1-score, and confusion matrices.

Use Grad-CAM and SHAP for explainability and interpretation.

Milestone 4 — Deployment & Reporting (Weeks 7–8)

Develop Streamlit UI to upload images and perform scanner identification with tamper detection.

Use calibrated SVMs with domain/type-specific thresholds.

Document system architecture, model comparisons, and provide demo screenshots.

Deliver final source code, trained model artifacts, and deployment instructions.

Methods
Residual Preprocessing

Convert to grayscale, resize to 256×256.

Perform Haar DWT denoising by zeroing detail bands and applying inverse transform.

Compute residual: Residual = Original - Denoised.

Feature Extraction

27-D handcrafted feature vector for hybrid model:

11 correlations with stored scanner fingerprints

6 FFT radial energy features

10 uniform LBP histogram bins

Tampering features:

Patch-level (22-D): LBP + FFT + residual + FFT resample features

Image-level (18-D): LBP + FFT + contrast statistics averaged across patches

Models

Scanner Identification: Dual-input hybrid CNN (residual image + 27-D feature vector).

Tampering Detection: Calibrated SVMs for patch-level and paired comparisons.

Top-k patch aggregation and adaptive thresholding for final tamper decision.

Installation

Requirements:

Python 3.8 or higher

Install dependencies:

pip install -r requirements.txt


Model artifact placement:
Place trained models and configuration files alongside app.py, or update paths as needed:

Scanner ID:

scanner_hybrid.keras

hybrid_label_encoder.pkl

hybrid_feat_scaler.pkl

scanner_fingerprints.pkl

fp_keys.npy

Tamper Detection (Image-level):

image_scaler.pkl

image_svm_sig.pkl

image_thresholds.json

Tamper Detection (Patch-level):

patch_scaler.pkl

patch_svm_sig_calibrated.pkl

thresholds_patch.json

Paired Tamper Detection:

pair_scaler.pkl

pair_svm_sig.pkl

pair_thresholds_topk.json

Usage

Run the Streamlit application:

streamlit run app.py


Upload supported image formats: TIFF, PNG, JPG

Output includes:

Predicted scanner model with confidence score.

Tamper detection result (clean/tampered) with probability and threshold info.

Debug details including domain, feature dimensions, and scaler information.

Evaluation

Scanner Identification: ~93% accuracy across 11 scanner classes.

Tampering Detection:

Patch-level AUC ≈ 0.93

Paired model AUC ≈ 0.81 for copy-move, retouch, and splicing tampering.

Thresholds tuned per domain and tamper type for robust deployment.

Dataset

Collected scans from multiple scanners and verified online repositories.

Organized manifests with file paths, class labels, tamper types, and page IDs.

Converted document PDFs to TIFF format for preprocessing and training.

Repository Structure
TraceFinder/
│
├── app.py                      # Streamlit app entrypoint
├── requirements.txt            # Python dependencies
│
├── notebooks/                  # Training and evaluation notebooks
├── artifacts/                  # Pretrained models, scalers, encoders
├── manifests/                  # Dataset manifest CSVs
└── data/                       # Raw and processed datasets

Acknowledgments

Inspired by research in forensic imaging and source identification.

Built using open-source ML and image processing libraries.

Thanks to mentors and contributors for support and guidance.

License

This project is released under the MIT License.
Please respect dataset license terms and provide appropriate credit when reusing or modifying the code.

Citation

TraceFinder: Scanner Identification and Tampering Detection System (2025).
Developed as part of the Infosys Springboard Virtual Internship 6.0.
