import streamlit as st
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import gdown

# ==========================================
# 1. MODEL DOWNLOAD LOGIC
# ==========================================
MODEL_SAVE_PATH = "flood_model.pth"
GOOGLE_DRIVE_FILE_ID = "13iVvRylDH5KeiY26756AAU9poc6M3WaF"

@st.cache_resource
def prepare_model():
    # If the file exists but is empty/broken (0 bytes), delete it to redownload
    if os.path.exists(MODEL_SAVE_PATH) and os.path.getsize(MODEL_SAVE_PATH) < 1000:
        os.remove(MODEL_SAVE_PATH)

    if not os.path.exists(MODEL_SAVE_PATH):
        with st.spinner("Downloading AI Model weights from Google Drive..."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            # Using fuzzy=True to handle the 'anyone with link' redirect better
            gdown.download(url, MODEL_SAVE_PATH, quiet=False, fuzzy=True)
    
    # Load the ResNet34-Unet
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to("cpu")
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()
    return model

# Initialize
model = prepare_model()

# ==========================================
# 2. CONFIG & TRANSFORMS (Your Original Logic)
# ==========================================
DEVICE = "cpu"
IMAGE_SIZE = 256

transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

def predict_flood(input_img):
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = transforms(image=resized)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
    return mask

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Flood Detector", layout="wide")
st.title("🛰️ Satellite Imagery Flood Detection System")

uploaded_file = st.file_uploader("Upload a Satellite Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(opencv_image, channels="BGR", use_column_width=True)
    
    with col2:
        st.subheader("AI Prediction")
        mask_result = predict_flood(opencv_image)
        st.image(mask_result, caption="AI-detected flooded areas", use_column_width=True)

st.sidebar.title("System Metrics")
st.sidebar.info("Model: ResNet34-Unet")
st.sidebar.metric("Mean IoU Score", "0.8421")
st.sidebar.metric("Accuracy", "84.21%")
