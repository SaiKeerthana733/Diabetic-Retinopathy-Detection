# Diabetic Retinopathy Detection using Deep Learning

## 📌 Introduction
Diabetic Retinopathy (DR) is a diabetes complication that can cause blindness. Early detection is critical. This project uses deep learning to classify retinal images into 5 DR stages.

## 🎯 Objectives
- Build a MobileNetV2 model for DR detection.
- Add Grad-CAM for explainability.
- Deploy as a Streamlit web app.

## 📊 Dataset
- Source: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- Classes: No DR, Mild, Moderate, Severe, Proliferative.

## ⚙️ Methodology
1. Preprocessing: resize, normalize, augment images.
2. Model: MobileNetV2 pretrained on ImageNet, fine-tuned for DR.
3. Explainability: Grad-CAM heatmaps to highlight important retinal regions.
4. Deployment: Streamlit app for interactive predictions.

## 🚀 How to Run
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app.py





## 📷 Screenrecord
[🎥 Watch the app screen recording](screenrecord/app_screen_recording.mp4)

### Streamlit App
![App Screen recording](screenrecord/app_screen_recording.mp4)




