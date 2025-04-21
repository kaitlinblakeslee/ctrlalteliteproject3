# Pneumonia Detection Using Chest X-rays

## Project Overview

This project uses deep learning to detect pneumonia in chest X-ray images. We leverage the RSNA Pneumonia Detection Challenge dataset provided on [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge). The model aims to classify pneumonia presence with high accuracy using transfer learning with pre-trained convolutional neural networks (CNNs).

## Objective

- Build a pneumonia classification model using a pre-trained CNN (e.g., ResNet50)
- Achieve a classification accuracy of **≥ 90%**
- Evaluate the model using **F1 score**, **AUC-ROC**, and **accuracy**
- Explore key dataset characteristics through EDA

## Dataset Summary

- **Source**: RSNA Pneumonia Detection Challenge (Kaggle)
- **Images**: ~30,000 chest X-rays in DICOM format
- **Metadata**:
  - Patient sex, age, projection view (PA/AP) in DICOM tags
  - Annotation labels provided in `stage_2_train_labels.csv`
- **Classes**:
  - `0`: No pneumonia
  - `1`: Pneumonia present

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/kaitlinblakeslee/ctrlalteliteproject3.git
cd ctrlalteliteproject3
```

### 2. Install Required Libraries
```bash
pip install pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
pydicom
opencv-python
jupyter
```
### 3. Download Data
Download the following files from Kaggle using this link: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview
- stage_2_train_images.zip
- stage_2_test_images.zip
- stage_2_train_labels.csv
  
Unzip the image folders into the root directory:
ctrlalteliteproject3/
├── stage_2_train_images/
├── stage_2_test_images/
├── stage_2_train_labels.csv
├── 01_EDA.ipynb
├── 02_Modeling_Pneumonia_Detection.ipynb
├── requirements.txt
└── README.md

### 4. Run Analysis Notebooks
1) FinalEDA.ipynb – for exploratory data analysis
2) FinalAnalysis.ipynb – for model training and evaluation

### 5. Evaluation Metric
- Accuracy

### 6. Acknowledgements
- RSNA and Kaggle for the dataset
- TensorFlow and Keras for deep learning libraries
