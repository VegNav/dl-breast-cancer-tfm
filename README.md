# TFM: Classification of Anomalies in Mammograms Using Deep Learning: A From-Scratch Approach
Deployment of classification models to be trained to classify masses anomalies in mammograms in bening or malignant.

This repository allows you to preprocess mammogram images and train up to 10 different classification models to compare their performance.
The script organize_data.py transforms the raw and disorganized CBIS-DDSM dataset into a structured format tailored for the binary classification task (benign vs malignant).

To use the organize_data.py script, you must first download the CBIS-DDSM dataset in DICOM format. This dataset is available on the official page of the Cancer Imaging Archive.
Note that downloading the dataset requires the external tool NBIA Data Retriever, which is also provided by TCIA.

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
