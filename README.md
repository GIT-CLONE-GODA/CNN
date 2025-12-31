

# 🧬 Skin Cancer Lesion Segmentation

A complete **deep learning project for skin lesion segmentation** using **PyTorch**, trained on the **HAM10000 dataset**, with deployment through **Streamlit** and **Gradio**.

---

## 📌 Overview

This project aims to segment skin lesions from dermoscopic images to support **skin cancer analysis**.
It implements and compares two advanced segmentation architectures and provides interactive interfaces for inference and visualization.

---

## 🧠 Models Used

* **Attention U-Net**
  Enhances segmentation accuracy using spatial attention on skip connections.

* **TransUNet Skip**
  CNN-based encoder with transformer-inspired bottleneck and skip connections.

---

## 📊 Dataset

* **HAM10000 (Human Against Machine)**
* Paired dermoscopic images and binary lesion masks
* Total samples: **5210**

  * Training: **4168**
  * Validation: **1042**

---

## 🔄 Preprocessing & Augmentation

Implemented using **Albumentations**:

* Resize to `256 × 256`
* Horizontal & Vertical Flip
* Random Rotation
* ImageNet Normalization
* Tensor Conversion

---

## 📉 Loss Function

A combined loss for robust segmentation:

```
Loss = Dice Loss + Binary Cross Entropy (BCE)
```

---

## 🏋️ Training Details

* Optimizer: Adam
* Learning Rate: `1e-4`
* Batch Size: `16`
* Epochs: `15`
* Best model saved based on validation loss

Saved weights:

* `best_attention_unet.pth`
* `best_transunet_skip.pth`

---

## 🌐 Deployment

### 🔹 Streamlit App

Interactive dashboard to:

* Upload an image
* Choose a model
* View original image, predicted mask, and overlay

Run:

```bash
streamlit run app.py
```

---

### 🔹 Gradio Interface

Quick demo interface with:

* Image upload
* Model selection
* Mask prediction
  Can be shared via public link or deployed on Hugging Face Spaces.
  ![Uploading WhatsApp Image 2025-12-29 at 4.48.26 PM.jpeg…]()


---

## 📁 Project Structure

```
.
├── Advanced_Medical_Image_Segmentation.ipynb
├── app.py
├── models/
│   ├── AttentionUNet.py
│   ├── TransUNetSkip.py
│   └── __init__.py
├── best_attention_unet.pth
├── best_transunet_skip.pth
├── requirements.txt
└── README.md
```

---

## 🎯 Applications

* Skin cancer research
* Medical image segmentation
* AI healthcare projects
* Graduation & portfolio projects

---
ML Engineer – Computer Vision & Deep Learning
Egypt 🇪🇬

---


