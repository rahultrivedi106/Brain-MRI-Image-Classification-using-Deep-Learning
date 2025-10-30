
# 🧠 Brain MRI Image Classification (TensorFlow)

**Author:** Rahul Trivedi  
_Independent Researcher, M.Tech_  
📩 rahultrivedi106@gmail.com  

---

## 📘 Overview
This repository presents a **TensorFlow-based Convolutional Neural Network (CNN)** for automatic **Brain MRI image classification**, distinguishing between *Ischemic* and *Hemorrhagic* stroke types.  
A lightweight CNN model has been implemented and trained on a **synthetic Kaggle-inspired MRI dataset**, provided for demonstration and reproducibility.

---

## 📂 Directory Structure
```
Brain-MRI-Classification-TF-Final/
│
├── src/                  # Model, evaluation & preprocessing scripts
│   ├── train_tf.py
│   ├── evaluate_tf.py
│   └── preprocess.py
│
├── data/                 #  MRI dataset (demo)
│   └── raw/
│       ├── train/
│       │   ├── ischemic/
│       │   └── hemorrhagic/
│       └── val/
│           ├── ischemic/
│           └── hemorrhagic/
│
figures/
├── model_architecture.png
├── training_curves.png
├── Ischemic_Stroke.jpg
├── Hemorrhagic_stroke.jpeg
├── Ischemic_pred_labeled.png
├── Hemorrhagic_pred_labeled.png
│
├── notebooks/            # Jupyter notebook (demo)
├── experiments/          # Model outputs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧩 Model Architecture
A simple CNN built using Keras sequential API.

![Model Architecture](Brain_Mri_image_classification/figures/model_architecture.png)

**Architecture Summary**
- Input: 224×224 grayscale MRI image  
- 3 × Convolution + MaxPooling blocks  
- Flatten → Dense(256) → Dropout(0.4)  
- Output: Softmax (2 classes)

---

## 🧠 Sample MRI Input

### 🩺 Ischemic Stroke
<p align="center">
  <img src="Brain_Mri_image_classification/figures/Ischemic Stroke.jpg" alt="Ischemic Stroke MRI" width="60%">
</p>

### 🧩 Hemorrhagic Stroke
<p align="center">
  <img src="Brain_Mri_image_classification/figures/Hemorrhagic stroke.jpeg" alt="Hemorrhagic Stroke MRI" width="60%">
</p>

Each MRI scan is preprocessed (grayscale, 224×224, normalized) before training.

---

## ⚙️ How to Run Locally

### 1️⃣ Create and activate virtual environment
```bash
python -m venv venv
# For Linux/Mac
source venv/bin/activate
# For Windows
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train model
```bash
python src/train_tf.py --epochs 10 --batch_size 8
```

Model weights will be saved to:
```
experiments/tf_model.h5
```

### 4️⃣ Evaluate model
```bash
python src/evaluate_tf.py
```

---

## 📊 Training Curves

![Training Curves](Brain_Mri_image_classification/figures/training_curves.png)

*Training accuracy and validation accuracy show consistent convergence.*

---

## 🧾 Prediction Output Example

<p align="center">
  <img src="Brain_Mri_image_classification/figures/Ischemic_pred_labeled.png" alt="Predicted Ischemic Stroke" width="45%">
  <img src="Brain_Mri_image_classification/figures/Hemorrhagic_pred_labeled.png" alt="Predicted Hemorrhagic Stroke" width="45%">
</p>

<p align="center"><b>Pred: Ischemic (0.86)</b> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>Pred: Hemorrhagic (0.89)</b></p>

*Model accurately distinguishes between stroke types based on MRI input.*

---

## 💾 Dataset Information
- Dataset included: **Synthetic MRI dataset (Ischemic vs Hemorrhagic)**  
- Inspired by the Kaggle public dataset:  
  [_Brain MRI Images for Brain Tumor Detection_](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
```

---

## 📈 Results Summary

| Metric | Value |
|--------|--------|
| Training Accuracy | ~92% |
| Validation Accuracy | ~88–90% |
| Loss | Stable (0.2–0.3) |

---

## 🧭 Pipeline Flow

**Data Loading → Preprocessing → Model Training → Validation → Evaluation → Visualization**

![Flowchart](figures/model_architecture.png)

---

## 🧱 Dependencies
```
tensorflow==2.12.0
numpy
pillow
matplotlib
scikit-learn
opencv-python
tqdm
```

---

## 📜 License
MIT License © 2025 Rahul Trivedi

---

## 📧 Contact
**Rahul Trivedi**  
📩 rahultrivedi106@gmail.com  
_Independent Researcher, M.Tech_

---
