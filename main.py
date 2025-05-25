from pathlib import Path
from source.organize_data import convert_recursive, split_data
from source.preprocess_data import *
from models.ResNet18_CFT import train_and_eval_resnet18_cft

def clean_path(text):
    return Path(text.strip().strip("'").strip('"'))

organize_data = input("Do you want to organize the data? (0-no, 1-yes): ")

if organize_data == "1":
    dicom_path = clean_path(input("Enter the path to the DICOM file: "))
    png_path = clean_path(input("Enter the path where the PNG image will be saved: "))

    convert_recursive(dicom_path, png_path)

    metadata_file_path = clean_path(input("Enter the path to the metadata CSV file: "))
    train_csv_path = clean_path(input("Enter the path to the training CSV file: "))
    test_csv_path = clean_path(input("Enter the path to the testing CSV file: "))
    image_base_path = clean_path(input("Enter the base path where the images are stored: "))
    output_dir = clean_path(input("Enter the directory where the organized dataset will be saved: "))

    split_data(metadata_file_path, train_csv_path, test_csv_path, image_base_path, output_dir)

preprocess = input("Do you want to preprocess the data? (0-no, 1-yes): ")

if preprocess == "1":
    original_base_dir = clean_path(input("Enter the path to the original dataset directory: "))
    cropped_base_dir = clean_path(input("Enter the path to the directory where cropped images will be saved: "))

    crop_dataset(original_base_dir, cropped_base_dir)

    drop_annotations_dir = clean_path(input("Enter the path to the directory containing images without annotations: "))

    drop_annotations_in_dataset(cropped_base_dir, drop_annotations_dir)

    blurred_dir = clean_path(input("Enter the path to the directory to contain the blurred images: "))

    apply_blur_to_dataset(drop_annotations_dir, blurred_dir)

    clahe_dir = clean_path(input("Enter the path to the directory to contain the CLAHE images: "))

    apply_clahe_to_dataset(blurred_dir, clahe_dir)

    augment_dir = clean_path(input("Enter the path to the directory to contain the augmented images: "))
    
    augment_dataset(clahe_dir, augment_dir)


train = input("Do you want to train a model? (0-no, 1-yes): ")

if train == "1":
    dataset_dir = clean_path(input("Enter the path to the dataset directory: "))

    train_and_eval_resnet18_cft(dataset_dir)