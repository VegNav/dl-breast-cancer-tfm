from pathlib import Path
from source.organize_data import convert_recursive, split_data
from source.preprocess_data import *
from models.ResNet18_OPW import eval_resnet18_opw
from models.ResNet18_PFT import train_and_eval_resnet18_pft
from models.ResNet18_CFT import train_and_eval_resnet18_cft
from models.DenseNet121_OPW import eval_densenet121_opw
from models.DenseNet121_PFT import train_and_eval_densenet121_pft
from models.DenseNet121_CFT import train_and_eval_densenet121_cft
from models.EfficientNetB3_OPW import eval_efficientnetB3_opw
from models.EfficientNetB3_CFT import train_and_eval_efficientnetB3_cft
from models.EfficientNetB3_Optuna import train_and_eval_efficientnetB3_optuna
from models.MammoCNN import train_and_eval_mammocnn


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

    model_choice = input("Choose a model to train (1- ResNet18, 2- DenseNet121, 3- EfficientNetB3, 4- MammoCNN): ")

    if model_choice == "1":
        resnet_choice = input("Choose a ResNet18 training method (1- OPW, 2- PFT, 3- CFT): ")
        if resnet_choice == "1":
            eval_resnet18_opw(dataset_dir)
        elif resnet_choice == "2":
            train_and_eval_resnet18_pft(dataset_dir)
        elif resnet_choice == "3":
            train_and_eval_resnet18_cft(dataset_dir)
    elif model_choice == "2":
        densenet_choice = input("Choose a DenseNet121 training method (1- OPW, 2- PFT, 3- CFT): ")
        if densenet_choice == "1":
            eval_densenet121_opw(dataset_dir)
        elif densenet_choice == "2":
            train_and_eval_densenet121_pft(dataset_dir)
        elif densenet_choice == "3":
            train_and_eval_densenet121_cft(dataset_dir)
    elif model_choice == "3":
        efficientnet_choice = input("Choose an EfficientNetB3 training method (1- OPW, 2- CFT, 3- Optuna): ")
        if efficientnet_choice == "1":
            eval_efficientnetB3_opw(dataset_dir)
        elif efficientnet_choice == "2":
            train_and_eval_efficientnetB3_cft(dataset_dir)
        elif efficientnet_choice == "3":
            train_and_eval_efficientnetB3_optuna(dataset_dir)
    elif model_choice == "4":
        train_and_eval_mammocnn(dataset_dir)
    else:
        print("Invalid model choice. Please choose a valid option (1-4).")
else:
    print("No model training selected. Exiting the program.")