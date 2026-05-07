import streamlit as st
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 1. SETTINGS & CONFIG (From your original code)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_SAVE_PATH = "flood_model.pth"

st.set_page_config(page_title="AI Flood Detection", layout="wide")
st.title("🛰️ Satellite Imagery Flood Detection System")

# ==========================================
# 2. DATASET LOGIC (Your exact original class)
# ==========================================
class FloodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
@st.cache_resource
def load_trained_model():
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to(DEVICE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        st.sidebar.success("✅ Model weights loaded!")
    else:
        st.sidebar.warning("⚠️ flood_model.pth not found. System running in training mode.")
    return model

model = load_trained_model()

# ==========================================
# 4. PREDICTION LOGIC (Your original predict_flood)
# ==========================================
transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

def predict_flood(input_img):
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
    return mask

# ==========================================
# 5. STREAMLIT UI (Web Version)
# ==========================================
tab1, tab2 = st.tabs(["🔍 Predict New Image", "📊 System Info"])

with tab1:
    uploaded_file = st.file_uploader("Upload Satellite Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(opencv_image, channels="BGR", caption="Original Image")
        
        with col2:
            mask_result = predict_flood(opencv_image)
            st.image(mask_result, caption="AI Detected Flood Area (Blue/White Mask)")

with tab2:
    st.info("Model: ResNet34-Unet | Input Size: 256x256")
    # Original Accuracy Log you provided
    st.write("🎯 FINAL MEAN IoU SCORE: 0.8421")
    st.write("📈 MODEL ACCURACY: 84.21%")
