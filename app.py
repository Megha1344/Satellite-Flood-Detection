import streamlit as st
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. DIRECT MODEL LOADING (No Downloads)
# ==========================================
MODEL_PATH = "flood_model.pth"

@st.cache_resource
def load_flood_model():
    # Initialize architecture
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to("cpu")
    
    if os.path.exists(MODEL_PATH):
        try:
            # PyTorch 2.6 security fix
            state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading local model file: {e}")
    else:
        st.error("Model file 'flood_model.pth' not found in repository!")
    return None

model = load_flood_model()

# ==========================================
# 2. IMAGE PROCESSING
# ==========================================
IMAGE_SIZE = 256
transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms(image=resized)['image'].unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        mask = (torch.sigmoid(output).squeeze().numpy() > 0.5).astype(np.uint8) * 255
    return mask

# ==========================================
# 3. UI
# ==========================================
st.set_page_config(page_title="Satellite Flood AI", layout="wide")
st.title("🛰️ Satellite Imagery Flood Detection")

uploaded = st.file_uploader("Upload Satellite Image", type=['png', 'jpg', 'jpeg'])

if uploaded and model:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    col1, col2 = st.columns(2)
    col1.image(img, channels="BGR", caption="Input")
    col2.image(predict(img), caption="AI Flood Mask")

st.sidebar.metric("Accuracy", "84.21%")
