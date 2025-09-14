import numpy as np
import cv2
import pywt

IMG_SIZE = (256, 256)

def preprocess_residual_from_array(image_np: np.ndarray) -> np.ndarray:
    """
    Matches the Colab inference pipeline:
    grayscale -> resize 256x256 -> float32 [0,1] ->
    Haar DWT denoise (zero detail bands) -> residual = img - denoised
    """
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    elif image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_np = image_np[..., 0]

    img = cv2.resize(image_np, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")

    residual = (img - den).astype(np.float32)
    return residual
