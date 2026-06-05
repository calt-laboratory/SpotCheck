# SpotCheck
**Android + PyTorch App for Benign/Malign Skin Lesion Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.8+-purple.svg)](https://kotlinlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)

---

## **Overview**
Mobile app to **classify skin moles as benign or malignant** using:
- **Dermatoscope/microscope-attached phone** for high-res images.
- **PyTorch (EfficientNetV2)** model trained on **HAM10000 + ISIC 2020**.
- **FastAPI backend** hosted on **Hetzner Cloud (Germany)** for GDPR compliance.
- **Kotlin (Android Studio)** frontend with **CameraX**.

**Disclaimer**: *For educational purposes only. NOT a medical diagnosis. Consult a dermatologist.*

---

## **Architecture**
```
[Android App (Kotlin + CameraX)]
       |
       +-- (HTTP + Image Upload)
[FastAPI Backend (PyTorch + ONNX)] --> [EfficientNetV2 Model]
       |
       v
[Hetzner Cloud Server (Germany)]
```

---

## **Roadmap**

| Phase | Task | Status |
|-------|------|--------|
| **Model** | Train EfficientNetV2 on HAM10000/ISIC | ⬜ |
| **Backend** | FastAPI + ONNX Runtime + Docker | ⬜ |
| **App** | Kotlin + CameraX + Ktor Client | ⬜ |
| **Deploy** | Hetzner Cloud (Nuremberg, GDPR-compliant) | ⬜ |

---

## **Tech Stack**

| Component | Technology |
|-----------|------------|
| **Model Training** | PyTorch + EfficientNetV2 + MONAI |
| **Model Format** | ONNX (for production) |
| **Backend** | FastAPI + Uvicorn + Docker |
| **Frontend** | Kotlin + Jetpack Compose + CameraX + Ktor Client |
| **Server** | Hetzner Cloud (Ubuntu + Docker) |
| **Data** | HAM10000 + ISIC 2020 |


