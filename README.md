# Driver Behaviour Detection Using Deep Learning

A computer vision project that detects distracted driver behaviour from images using deep learning. The project compares three custom Convolutional Neural Networks (CNNs) with a fine-tuned ResNet18 model to identify the most effective approach for driver behaviour classification.

---

## Overview

Driver distraction is one of the leading causes of road traffic accidents worldwide. Detecting unsafe driving behaviour automatically can support Advanced Driver Assistance Systems (ADAS), improve road safety, and assist in monitoring driver attention.

This project develops an image classification system capable of recognising different driver behaviours from a single image. Multiple deep learning models were implemented and compared to determine the most accurate and reliable solution.

To ensure realistic evaluation, the dataset was split using a **driver-aware strategy**, meaning images from the same driver never appeared in both the training and testing datasets. This measures how well the models generalise to completely unseen drivers rather than memorising individual appearances.

---

## Dataset

The project uses the **State Farm Distracted Driver Detection** dataset.

The original dataset contains ten behaviour classes. These were merged into four broader categories to simplify classification while maintaining meaningful distinctions between different driving behaviours.

| Behaviour Category | Original Classes |
|-------------------|------------------|
| Safe / Talking | Safe Driving, Talking to Passenger |
| Using Phone | Texting (Left & Right), Phone Call (Left &Right) |
| Distracted Driving | Operating Radio, Drinking |
| Reaching / Grooming | Reaching Behind, Hair & Makeup |

A **GroupShuffleSplit** strategy was used to ensure each driver only appears in one dataset partition.

---

## Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pillow
- Flask
- ipywidgets

---

## Models

### Baseline CNN
A simple convolutional neural network used as the baseline for performance comparison.

### Improved CNN v2
A lightweight CNN incorporating batch normalisation, dropout, and class weighting.

### Improved CNN v3
A deeper CNN architecture with additional convolutional layers and label smoothing for improved feature extraction.

### Fine-Tuned ResNet18
A ResNet18 model pre-trained on ImageNet and fine-tuned on the driver behaviour dataset. This achieved the highest overall performance.

---

## Results

| Model | Test Accuracy | Macro F1 |
|------|------:|------:|
| Baseline CNN | 54.77% | 0.5160 |
| Improved CNN v2 | 40.69% | 0.4074 |
| Improved CNN v3 | 69.32% | 0.7061 |
| **Fine-Tuned ResNet18** | **95.50%** | **0.9413** |

### Key Findings

- Transfer learning significantly outperformed CNNs trained from scratch.
- Driver-aware evaluation provided a more realistic measure of model performance.
- Increasing network depth improved the performance of custom CNN architectures.
- Pre-trained ImageNet features enabled excellent generalisation to previously unseen drivers.

---

## Features

- Driver behaviour image classification
- Four behaviour categories
- Driver-aware dataset splitting
- Comparison of multiple CNN architectures
- Transfer learning with ResNet18
- Training and validation visualisations
- Confusion matrix analysis
- Interactive prediction widget
- Flask web application for image classification

---

## Project Structure

```text
.
├── notebook.ipynb
├── app.py
├── models/
│   └── resnet18_finetuned.pth
├── templates/
│   └── index.html
├── static/
├── outputs/
│   ├── confusion_matrices/
│   ├── learning_curves/
│   └── model_comparison/
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/driver-behaviour-detection.git

cd driver-behaviour-detection

pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow flask ipywidgets
```

---

## Usage

### Train and Evaluate

Run the notebook sequentially to:

- Load and preprocess the dataset
- Create driver-aware train, validation, and test splits
- Train all deep learning models
- Evaluate model performance
- Compare architectures
- Generate predictions

### Run the Web Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5001
```

Upload an image and the application will predict the driver's behaviour.

---

## Future Improvements

- Real-time video-based behaviour recognition
- OpenCV webcam integration
- Vision Transformer (ViT) models
- MobileNet deployment for edge devices
- Temporal modelling using video sequences
- Integration with driver monitoring systems

---

## Author

**Aayosha Nepali**
