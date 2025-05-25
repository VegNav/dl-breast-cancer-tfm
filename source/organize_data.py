import os
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split



# PNG conversion
def dicom_to_png(dicom_path, output_path):
    """
    Converts a DICOM file to a PNG image and saves it to the specified output path.
    Args:
        dicom_path (str): Path to the input DICOM file.
        output_path (str): Path where the output PNG image will be saved.
    """
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array

    # Normalización a 0-255
    image = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    image = (image * 255).astype(np.uint8)

    # Guardar como PNG
    im = Image.fromarray(image)
    im.save(output_path)

def convert_recursive(input_root, output_root):
    """
    Recursively walks through the input directory, converts DICOM files to PNG,
    and saves them in the output directory while preserving the directory structure.
    Args:
        input_root (str): Root directory containing DICOM files.
        output_root (str): Root directory where PNG files will be saved.
    """
    for dirpath, dirnames, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                input_path = os.path.join(dirpath, filename)

                # To reproduce structure
                relative_path = os.path.relpath(dirpath, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_filename = filename.replace(".dcm", ".png")
                output_path = os.path.join(output_dir, output_filename)

                dicom_to_png(input_path, output_path)


# Train/val/test split
def split_data(metadata_file_path, train_csv_path, test_csv_path, image_base_path, output_dir):
    """
    Organize mammogram images into train, validation, and test sets based on metadata and pathology.
    CSV files are provided from the CBIS-DDSM dataset, which contains information about the images.
    Args:
        metadata_file_path (str): Path to the metadata CSV file.
        train_csv_path (str): Path to the training CSV file.
        test_csv_path (str): Path to the testing CSV file.
        image_base_path (str): Base path where the images are stored.
        output_dir (str): Directory where the organized dataset will be saved.
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for split in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split, 'benign'), exist_ok=True)
        os.makedirs(os.path.join(split, 'malignant'), exist_ok=True)

    # Load csv files
    metadata_df = pd.read_csv(metadata_file_path)
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Filtrar por tipo y patología
    train_df = train_df[(train_df['abnormality type'] == 'mass') & (train_df['pathology'].isin(['MALIGNANT', 'BENIGN']))]
    test_df = test_df[(test_df['abnormality type'] == 'mass') & (test_df['pathology'].isin(['MALIGNANT', 'BENIGN']))]
    all_cases_df = pd.concat([train_df, test_df])

    # Copiar imágenes a carpetas train/test
    for _, row in tqdm(all_cases_df.iterrows(), total = len(all_cases_df), desc="Organizing images"):
        patient_id = row['patient_id']
        pathology = row['pathology'].lower()
        series_desc = row['image file path'].split('/')[0]

        metadata_row = metadata_df[metadata_df['Subject ID'] == series_desc]

        if not metadata_row.empty:
            image_path = metadata_row['File Location'].values[0]
            full_image_path = os.path.abspath(os.path.join(
                image_base_path,
                image_path.replace('./CBIS-DDSM/', ''),
                '1-1.png'
            ))

            if os.path.exists(full_image_path):
                split_dir = train_dir if 'Train' in row['image file path'] else test_dir
                target_dir = os.path.join(split_dir, pathology)
                new_image_name = f"{series_desc}.png"
                target_path = os.path.join(target_dir, new_image_name)

                shutil.copy(full_image_path, target_path)
            else:
                print(f"[WARNING] Image not found: {full_image_path}")
        else:
            print(f"[WARNING] Metadata not found: {series_desc}")

    # Obtain roots from train set to make the split
    all_image_paths = []
    for pathology in ['benign', 'malignant']:
        path_folder = os.path.join(train_dir, pathology)
        for image_name in os.listdir(path_folder):
            all_image_paths.append(os.path.join(path_folder, image_name))

    # Train/val split
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2)

    for image_path in tqdm(val_paths, desc="Moving images to validation set"):
        target_subdir = 'benign' if 'benign' in image_path else 'malignant'
        target_dir = os.path.join(val_dir, target_subdir)
        shutil.move(image_path, target_dir)

