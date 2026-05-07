import streamlit as st
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. MODEL LOADING (Local Repository)
# ==========================================
# This must match the name of the file you upload via GitHub Desktop
MODEL_PATH = "flood_model.pth"

@st.cache_resource
def load_flood_model():
    # Initialize the same architecture used in training
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to("cpu")
    
    if os.path.exists(MODEL_PATH):
        try:
            # weights_only=False is required for PyTorch 2.6+ to load custom .pth files
            state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    else:
        st.error(f"Model file '{MODEL_PATH}' not found! Please ensure it was pushed via GitHub Desktop.")
    return None

# Initialize the model
model = load_flood_model()

# ==========================================
# 2. IMAGE PRE-PROCESSING
# ==========================================
IMAGE_SIZE = 256

# Standard ImageNet normalization used during your Colab training
transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(img):
    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Apply transforms and add batch dimension
    tensor = transforms(image=resized)['image'].unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        # Convert output to binary mask (Flooded vs Not Flooded)
        mask = (torch.sigmoid(output).squeeze().numpy() > 0.5).astype(np.uint8) * 255
    return mask

# ==========================================
# 3. STREAMLIT WEB UI
# ==========================================
st.set_page_config(page_title="Satellite Flood AI", layout="wide", page_icon="🛰️")

st.title("🛰️ Satellite Imagery Flood Detection System")
st.write("Professional Engineering Project: Deep Learning for Disaster Management")
st.divider()

uploaded_file = st.file_uploader("Upload a Satellite Image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    if model is None:
        st.error("Model is not initialized. Check if 'flood_model.pth' is in the repository.")
    else:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Satellite Image")
            st.image(opencv_img, channels="BGR", use_container_width=True)
        
        with col2:
            st.subheader("AI Flood Prediction")
            with st.spinner("Analyzing imagery..."):
                mask_result = predict(opencv_img)
                # Display the binary mask
                st.image(mask_result, caption="White = Flooded Areas | Black = Ground/Water", use_container_width=True)

# Sidebar Technical Details
st.sidebar.title("Model Dashboard")
st.sidebar.info("Architecture: ResNet34-Unet")
st.sidebar.metric("Mean IoU Score", "0.8421")
st.sidebar.metric("Accuracy", "84.21%")
st.sidebar.divider()
st.sidebar.write("Project by Megha Prathish")
