# 🩸 Leukemia Identification & Analysis System (LIAS)

An AI-powered Blood Cancer (Leukemia) Detection System developed using Deep Learning and Computer Vision.

---

## 📌 Overview

Leukemia Identification & Analysis System (LIAS) is an intelligent medical image analysis application that detects blood cancer (Leukemia) from microscopic blood smear images.

The system uses EfficientNetV2-B0 with transfer learning to classify blood smear images into four categories: Healthy, Acute Lymphoblastic Leukemia (ALL), Acute Myeloid Leukemia (AML), and Chronic Myeloid Leukemia (CML).

To improve interpretability, Grad-CAM explainable AI is integrated to highlight the image regions influencing the model's predictions. The project also includes an interactive Streamlit dashboard for real-time image analysis and automated report generation.

---

## 🚀 Features

- AI-based Blood Cancer Detection
- Four-Class Leukemia Classification
- EfficientNetV2 Deep Learning Model
- Image Preprocessing Pipeline
- Grad-CAM Explainable AI
- Streamlit Web Application
- Medical Report Generator
- Real-time Predictions

---

## 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Deep Learning | EfficientNetV2-B0 |
| Explainable AI | Grad-CAM |
| Web Framework | Streamlit |
| Image Processing | Pillow |
| Data Analysis | NumPy, Pandas |

---

## 📂 Project Structure

```text
blood-cancer-leukemia-detection-ai/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── model/
├── preprocessing/
├── report/
├── screenshots/
├── results/
└── docs/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/itssagraxen02/blood-cancer-leukemia-detection-ai.git

cd blood-cancer-leukemia-detection-ai

pip install -r requirements.txt

streamlit run app.py
```

---

## 📊 Model Workflow

```
Blood Smear Image
        │
        ▼
Image Preprocessing
        │
        ▼
EfficientNetV2 Model
        │
        ▼
Prediction
        │
        ▼
Grad-CAM Heatmap
        │
        ▼
Medical Report
```

---

## 📸 Screenshots

Add screenshots here after uploading them.

- Home Page
- Prediction Page
- Grad-CAM Result
- Dashboard

---

## 🔮 Future Improvements

- Cloud Deployment
- Mobile Application
- Docker Support
- Multi-Disease Detection
- REST API Integration

---

## 👨‍💻 Author

**Agrasen Chaudhary**

B.Tech Computer Science & Engineering

Shri Ramswaroop Memorial University

---

## 📄 License

MIT License
