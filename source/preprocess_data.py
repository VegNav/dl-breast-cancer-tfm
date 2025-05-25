import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

def crop_fixed_margin(image, margin_ratio=0.05):
    """
    Crop a fixed margin from the image.
    Args:
        image (numpy.ndarray): Input image to crop.
        margin_ratio (float): Ratio of the margin to crop from each side of the image."""
    h, w = image.shape[:2]

    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)

    cropped = image[margin_h:h-margin_h, margin_w:w-margin_w]

    return cropped



def crop_dataset(original_base_dir, cropped_base_dir):
    """
    Crop images in the dataset by a fixed margin and save them to a new directory.
    Args:
        original_base_dir (str): Path to the original dataset directory.
        cropped_base_dir (str): Path to the directory where cropped images will be saved.
    """
    # Walk through the dataset
    sets = ['train', 'val', 'test']
    categories = ['benign', 'malignant']
    for set_name in sets:
        for category in categories:
            input_dir = os.path.join(original_base_dir, set_name, category)
            output_dir = os.path.join(cropped_base_dir, set_name, category)
            os.makedirs(output_dir, exist_ok=True)

            image_names = os.listdir(input_dir)

            for img_name in tqdm(image_names, desc=f"{set_name}/{category}"):
                img_path = os.path.join(input_dir, img_name)
                save_path = os.path.join(output_dir, img_name)

                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Reading error in {img_path}")
                    continue

                # Apply crop
                cropped_img = crop_fixed_margin(img)

                # Save cropped image
                cv2.imwrite(save_path, cropped_img)


def drop_annotations(img_path, output_path):
    """
    Drop annotations from the mammography image by applying morphological operations.
    Args:
        img_path (str): Path to the input mammography image.
        output_path (str): Path to save the processed image without annotations.
    """
    # Read the images
    img = cv2.imread(img_path)
    hh, ww = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # AApply close morphology to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Apply open morplogy to separate the mamary from the annotations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Find the biggest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # Create a mask with the smallest contours out
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # Apply dilate morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)

    # Apply mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save the result
    cv2.imwrite(output_path, result)

def drop_annotations_in_dataset(input_dir, output_dir):
    sets = ['train', 'val', 'test']
    categories = ['benign', 'malignant']
    for set_name in sets:
        for category in categories:
            input_dir = os.path.join(input_dir, set_name, category)
            output_dir = os.path.join(output_dir, set_name, category)
            os.makedirs(output_dir, exist_ok=True)

            image_names = os.listdir(input_dir)

            for img_name in tqdm(image_names, desc=f"Dropping annotations. {set_name}/{category}"):
                img_path = os.path.join(input_dir, img_name)
                save_path = os.path.join(output_dir, img_name)

                # Leer imagen
                img = cv2.imread(img_path)

                drop_annotations(img_path, save_path)


def apply_blur_to_dataset(input_dir, output_dir):
    """
    Apply bilateral filter to images in the dataset to reduce noise while keeping edges sharp.
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the directory where processed images will be saved.
    """
    # Bilateral filter parameters
    d=5
    sigmaColor=90
    sigmaSpace=90

    sets = ['train', 'val', 'test']
    categories = ['benign', 'malignant']

    for set_name in sets:
        for category in categories:
            input_dir = os.path.join(input_dir, set_name, category)
            output_dir = os.path.join(output_dir, set_name, category)
            os.makedirs(output_dir, exist_ok=True)

            image_names = os.listdir(input_dir)

            for img_name in tqdm(image_names, desc=f"Applying blur. {set_name}/{category}"):
                img_path = os.path.join(input_dir, img_name)
                save_path = os.path.join(output_dir, img_name)

                # Read image
                img = cv2.imread(img_path)

                # Apply Gaussian Blur
                blurred_img = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )

                # Save blurred image
                cv2.imwrite(save_path, blurred_img)


def clahe_def(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast in the image.
    Args:
        img (numpy.ndarray): Input image to apply CLAHE.
    Returns:
        numpy.ndarray: Image after applying CLAHE.
    """
    # Convert grayscale if the image is in color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crate CLAHE object
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))

    # Apply CLAHE
    clahe_img = clahe_obj.apply(img)

    return clahe_img


def apply_clahe_to_dataset(input_dir, output_dir):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to images in the dataset.
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the directory where processed images will be saved.
    """
    sets = ['train', 'val', 'test']
    categories = ['benign', 'malignant']

    for set_name in sets:
        for category in categories:
            input_dir = os.path.join(input_dir, set_name, category)
            output_dir = os.path.join(output_dir, set_name, category)
            os.makedirs(output_dir, exist_ok=True)

            image_names = os.listdir(input_dir)

            for img_name in tqdm(image_names, desc=f"Applying CLAHE. {set_name}/{category}"):
                img_path = os.path.join(input_dir, img_name)
                save_path = os.path.join(output_dir, img_name)

                # Read image
                img = cv2.imread(img_path)

                # Apply CLAHE
                clahe_img = clahe_def(img)

                # Save processed image
                cv2.imwrite(save_path, clahe_img)
    

def augment_image(img):
    """
    Augment the input image with various transformations.
    Args:
        img (numpy.ndarray): Input image to augment.
    Returns:
        list: List of augmented images.
    """
    augmented_images = []

    # 1. Original
    augmented_images.append(img)

    # 2. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 3. Vertical flip 
    flipped_v = cv2.flip(img, 0)
    augmented_images.append(flipped_v)

    # 4. +10 rotation
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    augmented_images.append(rotated)

    # 5. -10 rotation
    rows, cols = img.shape[:2]
    M_inverse = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    rotated_inverse = cv2.warpAffine(img, M_inverse, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    augmented_images.append(rotated_inverse)

    return augmented_images

def augment_dataset(input_dir, output_dir):
    """
    Augment images in the dataset by applying various transformations.
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the directory where augmented images will be saved.
    """
    sets = ['train', 'val', 'test']
    categories = ['benign', 'malignant']
    
    for set_name in sets:
        for category in categories:
            input_dir = os.path.join(input_dir, set_name, category)
            output_dir = os.path.join(output_dir, set_name, category)
            os.makedirs(output_dir, exist_ok=True)

            image_names = os.listdir(input_dir)

            for img_name in tqdm(image_names, desc=f"Data augmentation. {set_name}/{category}"):
                img_path = os.path.join(input_dir, img_name)
                img = cv2.imread(img_path)

                # Apply augmentations only to training set
                if set_name == "train":
                    altered_imgs = augment_image(img)

                    base_filename, ext = os.path.splitext(img_name)

                    for idx, altered_img in enumerate(altered_imgs):
                        new_name = f"{base_filename}_aug{idx}{ext}"
                        save_path = os.path.join(output_dir, new_name)
                        cv2.imwrite(save_path, altered_img)

                else:
                    shutil.copy(img_path, output_dir)