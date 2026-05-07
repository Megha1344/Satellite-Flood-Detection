import streamlit as st
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import gdown

# ==========================================
# 1. AUTOMATIC MODEL DOWNLOAD (The Workaround)
# ==========================================
MODEL_SAVE_PATH = "flood_model.pth"
GOOGLE_DRIVE_FILE_ID = "13iVvRylDH5KeiY26756AAU9poc6M3WaF"

@st.cache_resource
def prepare_model():
    # If the model isn't there, download it
    if not os.path.exists(MODEL_SAVE_PATH):
        with st.spinner("Downloading AI Model weights (approx. 90MB)... Please wait."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            gdown.download(url, MODEL_SAVE_PATH, quiet=False)
    
    # Load the model exactly as per your original logic
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to("cpu") # Use CPU for Streamlit Cloud
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()
    return model

# Initialize the model
model = prepare_model()

# ==========================================
# 2. YOUR ORIGINAL SETTINGS & DATASET CLASS
# ==========================================
DEVICE = "cpu"
IMAGE_SIZE = 256

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
        
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

# ==========================================
# 3. YOUR ORIGINAL PREDICTION & METRIC LOGIC
# ==========================================
transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

def predict_flood(input_img):
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # Match your original preprocessing
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = transforms(image=resized)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
    return mask

# ==========================================
# 4. STREAMLIT WEB UI
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
        st.subheader("Original Image")
        st.image(opencv_image, channels="BGR", use_column_width=True)
    
    with col2:
        st.subheader("AI Prediction")
        mask_result = predict_flood(opencv_image)
        st.image(mask_result, caption="Blue/White areas indicate Flooding", use_column_width=True)

st.sidebar.title("Model Metrics")
st.sidebar.info("Model: ResNet34-Unet")
st.sidebar.metric("Mean IoU Score", "0.8421")
st.sidebar.metric("Accuracy", "84.21%")
