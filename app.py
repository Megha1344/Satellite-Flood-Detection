import streamlit as st
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import requests

# ==========================================
# 1. ROBUST MODEL DOWNLOAD (Final Version)
# ==========================================
MODEL_SAVE_PATH = "flood_model.pth"
GOOGLE_DRIVE_FILE_ID = "13iVvRylDH5KeiY26756AAU9poc6M3WaF"

@st.cache_resource
def prepare_model():
    # 1. Cleanup: If the file is an HTML error page (starts with '<'), delete it
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            with open(MODEL_SAVE_PATH, 'rb') as f:
                first_char = f.read(1)
                if first_char == b'<':
                    os.remove(MODEL_SAVE_PATH)
        except:
            pass

    # 2. Download: Using 'confirm=t' to bypass Google's large file warning
    if not os.path.exists(MODEL_SAVE_PATH) or os.path.getsize(MODEL_SAVE_PATH) < 10000:
        with st.spinner("Downloading AI Model weights from Google Drive..."):
            url = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=t'
            response = requests.get(url, stream=True)
            with open(MODEL_SAVE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    
    # 3. Initialization: Setup ResNet34-Unet structure
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to("cpu")
    
    # 4. Loading: Load weights with PyTorch 2.6 security bypass
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            # If load fails, delete the file so it retries on next refresh
            if os.path.exists(MODEL_SAVE_PATH):
                os.remove(MODEL_SAVE_PATH)
            
    model.eval()
    return model

# Initialize the model
model = prepare_model()

# ==========================================
# 2. IMAGE PRE-PROCESSING
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
# 3. WEB INTERFACE (Streamlit)
# ==========================================
st.set_page_config(page_title="AI Flood Detector", layout="wide")
st.title("🛰️ Satellite Imagery Flood Detection System")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a Satellite Image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Satellite Image")
        st.image(opencv_image, channels="BGR", use_container_width=True)
    
    with col2:
        st.subheader("AI Flood Segmentation")
        with st.spinner("Model is analyzing imagery..."):
            mask_result = predict_flood(opencv_image)
            # Displaying the result
            st.image(mask_result, caption="White areas represent predicted flooding", use_container_width=True)

# Sidebar Technical Details
st.sidebar.title("Model Dashboard")
st.sidebar.info("Model: ResNet34-Unet")
st.sidebar.metric("Mean IoU", "0.8421")
st.sidebar.metric("Accuracy", "84.21%")
st.sidebar.markdown("---")
st.sidebar.write("Developed for Satellite Flood Monitoring")
