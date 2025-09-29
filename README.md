# Lung-Cancer-Detection-Using-Transfer-Learning

## 📘 Project Overview

This project leverages Transfer Learning to develop a deep learning model capable of classifying lung tissue images into three categories:

- **Normal**
- **Lung Adenocarcinoma**
- **Lung Squamous Cell Carcinoma**

By utilizing pre-trained models, the project aims to enhance classification accuracy and reduce training time.

---

## 🧪 Technologies & Libraries Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - OpenCV
  - Scikit-learn
- **Platform**: Google Colab (for cloud-based execution)

---

## 📥 Dataset

The dataset comprises 5,000 images categorized into three classes:

- **Normal**
- **Lung Adenocarcinomas**
- **Lung Squamous Cell Carcinomas**

Each class contains 250 images, augmented to 5,000 images in total. The dataset is sourced from Kaggle.

---

## ⚙️ Model Architecture

The model employs a pre-trained Convolutional Neural Network (CNN) for feature extraction, followed by fine-tuning to adapt to the lung cancer classification task. This approach leverages the knowledge from models trained on large datasets like ImageNet, improving performance on the target task.

---

📈 Results

The model achieves high classification accuracy on the validation set, typically around 95%, demonstrating the effectiveness of Transfer Learning in medical image classification tasks. You can monitor:

Training Accuracy

Validation Accuracy

Loss Metrics

Confusion Matrix for Classification Performance
