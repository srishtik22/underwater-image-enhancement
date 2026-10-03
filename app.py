import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ================================================================
# PIPELINE CODE (Progress 1 + Progress 2)
# ================================================================

def to_gray(img_float):
    img_uint8 = (img_float * 255).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

def light_denoise(img_float):
    img_uint8 = (img_float * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=35, sigmaSpace=35)
    return denoised.astype(np.float32) / 255.0

def preprocess_image(pil_image, target_size=(512, 512)):
    img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return light_denoise(img)

def estimate_colour_cast(img_float):
    r_mean, g_mean, b_mean = [img_float[:, :, i].mean() for i in range(3)]
    deficit = max(0.0, (g_mean + b_mean) / 2.0 - r_mean)
    return float(min(deficit / 0.4, 1.0))

def _dark_channel(img_float, patch_size=15):
    min_channel = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)

def estimate_haze(img_float):
    return float(min(_dark_channel(img_float).mean() / 0.5, 1.0))

def estimate_contrast_loss(img_float):
    std = to_gray(img_float).std()
    return float(min(max(0.0, 1.0 - (std / 0.22)), 1.0))

def estimate_noise(img_float):
    gray = to_gray(img_float)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(min((gray - blurred).std() / 0.03, 1.0))

def estimate_detail_loss(img_float):
    gray = to_gray(img_float)
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    return float(min(max(0.0, 1.0 - (lap_var / 0.01)), 1.0))

def estimate_all(img_float):
    scores = {
        "colour_cast": estimate_colour_cast(img_float),
        "haze": estimate_haze(img_float),
        "contrast_loss": estimate_contrast_loss(img_float),
        "noise": estimate_noise(img_float),
        "detail_loss": estimate_detail_loss(img_float),
    }
    scores["dominant_degradation"] = max(scores, key=scores.get)
    return scores

def colour_correction(img_float, strength=1.0):
    result = img_float.copy()
    means = img_float.reshape(-1, 3).mean(axis=0)
    gray_mean = means.mean()
    gains = gray_mean / (means + 1e-6)
    blended_gains = 1.0 + strength * (gains - 1.0)
    for c in range(3):
        result[:, :, c] = img_float[:, :, c] * blended_gains[c]
    return np.clip(result, 0, 1)

def dehaze(img_float, strength=1.0, patch_size=15, omega=0.95, t0=0.1):
    dark = _dark_channel(img_float, patch_size)
    flat_dark = dark.flatten()
    flat_img = img_float.reshape(-1, 3)
    num_pixels = max(int(0.001 * flat_dark.size), 1)
    top_indices = np.argsort(flat_dark)[-num_pixels:]
    veiling_light = np.clip(flat_img[top_indices].mean(axis=0), 0.2, 1.0)
    normalized = img_float / veiling_light[None, None, :]
    transmission = np.clip(1 - omega * _dark_channel(normalized, patch_size), t0, 1.0)
    recovered = (img_float - veiling_light[None, None, :]) / transmission[:, :, None] + veiling_light[None, None, :]
    recovered = np.clip(recovered, 0, 1)
    result = img_float * (1 - strength) + recovered * strength
    return np.clip(result, 0, 1)

def contrast_enhancement(img_float, strength=1.0, clip_limit=2.0):
    img_uint8 = (img_float * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    effective_clip = 0.1 + strength * clip_limit
    clahe = cv2.createCLAHE(clipLimit=effective_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result_uint8 = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return result_uint8.astype(np.float32) / 255.0

def denoise(img_float, strength=1.0):
    img_uint8 = (img_float * 255).astype(np.uint8)
    d = max(3, int(3 + strength * 6))
    sigma = 20 + strength * 40
    denoised = cv2.bilateralFilter(img_uint8, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    return denoised.astype(np.float32) / 255.0

def sharpen(img_float, strength=1.0):
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=3)
    amount = strength * 1.5
    return np.clip(img_float + amount * (img_float - blurred), 0, 1)

DEFAULT_STRENGTHS = {
    "colour_correction": 0.7, "dehaze": 0.6, "contrast_enhancement": 0.5,
    "denoise": 0.4, "sharpen": 0.4,
}

def fixed_pipeline(img_float):
    result = img_float.copy()
    result = colour_correction(result, strength=DEFAULT_STRENGTHS["colour_correction"])
    result = dehaze(result, strength=DEFAULT_STRENGTHS["dehaze"])
    result = contrast_enhancement(result, strength=DEFAULT_STRENGTHS["contrast_enhancement"])
    result = denoise(result, strength=DEFAULT_STRENGTHS["denoise"])
    result = sharpen(result, strength=DEFAULT_STRENGTHS["sharpen"])
    return result

def compute_adaptive_strengths(degradation_scores):
    raw_strengths = {
        "colour_correction": degradation_scores["colour_cast"],
        "dehaze": degradation_scores["haze"],
        "contrast_enhancement": degradation_scores["contrast_loss"],
        "denoise": min(degradation_scores["noise"], 0.7),
        "sharpen": degradation_scores["detail_loss"] * (1 - 0.5 * degradation_scores["noise"]),
    }
    return {m: float(max(0.0, min(v, DEFAULT_STRENGTHS[m]))) for m, v in raw_strengths.items()}

def adaptive_pipeline(img_float):
    degradation_scores = estimate_all(img_float)
    strengths = compute_adaptive_strengths(degradation_scores)
    result = img_float.copy()
    result = colour_correction(result, strength=strengths["colour_correction"])
    result = dehaze(result, strength=strengths["dehaze"])
    result = contrast_enhancement(result, strength=strengths["contrast_enhancement"])
    result = denoise(result, strength=strengths["denoise"])
    result = sharpen(result, strength=strengths["sharpen"])
    return result, degradation_scores, strengths

def compute_uciqe(img_float):
    img_uint8 = (img_float * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    l_sorted = np.sort(l.flatten())
    con_l = l_sorted[int(0.99 * len(l_sorted))] - l_sorted[int(0.01 * len(l_sorted))]
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    sat_mean = hsv[:, :, 1].astype(np.float32).mean()
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * sat_mean)

# ================================================================
# DETECTION MODEL (loaded once, cached)
# ================================================================
@st.cache_resource
def load_detection_model():
    weights_path = hf_hub_download(repo_id="dronefreak/brackish-yolov8m", filename="best.pt")
    return YOLO(weights_path)

def detect(img_float, model, conf=0.25):
    img_uint8 = (img_float * 255).astype(np.uint8)
    results = model.predict(img_uint8, conf=conf, verbose=False)[0]
    annotated = results.plot()  # returns image with boxes drawn (BGR)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    num = len(results.boxes)
    mean_conf = float(results.boxes.conf.mean()) if num > 0 else 0.0
    return annotated_rgb, num, mean_conf

# ================================================================
# STREAMLIT APP
# ================================================================
st.set_page_config(page_title="Adaptive Underwater Image Enhancement", layout="wide")
st.title("🌊 Adaptive Underwater Image Enhancement")
st.write("Upload an underwater photo to see it enhanced two ways — a fixed baseline pipeline and an adaptive, degradation-aware pipeline — plus object detection on each version.")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    with st.spinner("Processing..."):
        img = preprocess_image(pil_image)
        fixed_result = fixed_pipeline(img)
        adaptive_result, degradation_scores, strengths = adaptive_pipeline(img)

        model = load_detection_model()
        raw_det_img, raw_n, raw_conf = detect(img, model)
        fixed_det_img, fixed_n, fixed_conf = detect(fixed_result, model)
        adaptive_det_img, adaptive_n, adaptive_conf = detect(adaptive_result, model)

    st.subheader("Degradation Diagnosis")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, degradation_scores.items()):
        if k != "dominant_degradation":
            col.metric(k.replace("_", " ").title(), f"{v:.2f}")
    st.write(f"**Dominant issue:** {degradation_scores['dominant_degradation'].replace('_', ' ').title()}")

    st.subheader("Enhancement + Detection Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Raw (Original)**")
        st.image(raw_det_img, use_container_width=True)
        st.write(f"Detections: {raw_n} | Avg confidence: {raw_conf:.2f}")
        st.write(f"UCIQE: {compute_uciqe(img):.1f}")

    with col2:
        st.markdown("**Fixed Pipeline**")
        st.image(fixed_det_img, use_container_width=True)
        st.write(f"Detections: {fixed_n} | Avg confidence: {fixed_conf:.2f}")
        st.write(f"UCIQE: {compute_uciqe(fixed_result):.1f}")

    with col3:
        st.markdown("**Adaptive Pipeline**")
        st.image(adaptive_det_img, use_container_width=True)
        st.write(f"Detections: {adaptive_n} | Avg confidence: {adaptive_conf:.2f}")
        st.write(f"UCIQE: {compute_uciqe(adaptive_result):.1f}")

    st.subheader("Adaptive Strengths Used")
    st.json(strengths)
else:
    st.info("Upload an image above to get started. Works best with real underwater photos (fish, coral, divers, etc.)")
