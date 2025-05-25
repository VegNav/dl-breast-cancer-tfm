# TFM: Classification of Anomalies in Mammograms Using Deep Learning: A From-Scratch Approach
Deployment of classification models to be trained to classify masses anomalies in mammograms in bening or malignant.

## About
This repository provides tools to preprocess mammogram images and train up to 10 different classification models for performance comparison. It was developed as part of the Master's Final Project in the Data Science Master's program at the Universitat Oberta de Catalunya (UOC).

## Installation
All the packages needed to run this project are available in the requirements.txt
```
pip install -r requirements.txt
```

## Scripts
The main.py script is structured as a guided questionnaire, prompting the user step-by-step through each phase of the project — from data organization and preprocessing to model training and evaluation. This design makes it easier to run specific stages interactively and ensures flexibility during experimentation.

The script organize_data.py transforms the raw and disorganized CBIS-DDSM dataset into a structured format tailored for the binary classification task (benign vs malignant).

To use the organize_data.py script, you must first download the CBIS-DDSM dataset in DICOM format. This dataset is available on the official page of the Cancer Imaging Archive. 
Note that downloading the dataset requires the external tool NBIA Data Retriever, which is also provided by TCIA.

After using the organize_data.py script, the dataset's folder architecture would be like this:

```
dataset/
├── train/
│   ├── benign/
│   └── malignant/
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```
The script `preprocess_data.py` includes all necessary preprocessing steps to prepare mammogram images for training. Specifically, it performs:

- **Border cropping**: Removes unnecessary borders around the images  
- **Annotation removal**: Eliminates radiologist markers and labels  
- **Gaussian blur**: Smooths the image to reduce noise  
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast  
- **Data augmentation**: Increases dataset variability through transformations

Finally, the models folder contains the code to train and evaluate 4 arquitectures, 3 of them with 3 variations: OPW (Oly Pretrained Weights. No layer is trained), PFT (Partial Fine Tuning. Final layers are trained) and CPT (Complete Fine Tuning. All layers are trained). The following models are able to train and evaluate:

+ **ResNet18_OPW**
+ **ResNet18_PFT**
+ **ResNet18_CFT**
+ **DenseNet121_OPW**
+ **DenseNet121_PFT**
+ **DenseNet121_CFT**
+ **EfficientNetB3_OPW**
+ **EfficientNetB3_CFT**
+ **EfficientNetB3_Optuna** - * Optuna was used to find the best parameters.
+ **MammoCNN** - Custom architecture.

I welcome contributions to improve the results; mistakes are just part of the path to success.


## Creator
This project has been fully created by Pedro Vega.