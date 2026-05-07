# Satellite-Flood-Detection
Satellite Imagery Flood Detection System
AI-Powered Disaster Management using Deep Learning
This repository contains a complete end-to-end pipeline for detecting flooded areas from satellite imagery. By leveraging the ResNet34-Unet architecture, the system performs pixel-level semantic segmentation to distinguish between land and water with high precision.

🚀 Project Overview
In disaster scenarios, rapid and accurate identification of flooded regions is critical for rescue operations. This project uses a U-Net based neural network to process satellite frames and generate binary masks highlighting affected areas.

Key Features
Architecture: ResNet34 encoder with a U-Net decoder for multi-scale feature extraction.

Preprocessing: Advanced augmentation via albumentations (Vertical/Horizontal flips, Normalization).

Accuracy: Optimized using Dice Loss to handle class imbalance in satellite data.

Web Dashboard: Live deployment on Streamlit for real-time image uploads and inference.

🛠️ Tech Stack
Framework: PyTorch

Library: Segmentation Models PyTorch (SMP)

Computer Vision: OpenCV, Albumentations

Deployment: Streamlit

Environment: Developed in Google Colab, deployed on Streamlit Cloud.

📈 Performance Metrics
The model was trained for 50 epochs and evaluated using the Intersection over Union (IoU) metric, which is the gold standard for segmentation tasks.

Final Mean IoU Score: 0.8421

Overall Model Accuracy: 84.21%

📁 Repository Structure
Plaintext
├── app.py                # Streamlit Web Application
├── flood_model.pth       # Trained ResNet34-Unet weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
💻 Setup & Installation
Clone the repository:

Bash
git clone https://github.com/Megha1344/Satellite-Flood-Detection.git
Install dependencies:

Bash
pip install -r requirements.txt
Run the app locally:

Bash
streamlit run app.py
