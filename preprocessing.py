import os
import sys
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from utils_preprocessing import create_csv_labels

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_DIR)

def run_preprocessing():
    
    dataset_path = PARENT_DIR
    
    image_path = os.path.join(dataset_path, 'dataset/img_resized')
    image_path_tens = os.path.join(dataset_path, 'dataset/img_tens')
    img_text_path = os.path.join(dataset_path, 'dataset/img_txt')
    json_path = os.path.join(dataset_path, 'dataset/MMHS150K_GT.json')
    GT_path = os.path.join(dataset_path, 'dataset/MMHS150K_Custom.csv')

    # Create the CSV file ==============================================================================
    print(f"INFO: Creating CSV file at {GT_path}")
    create_csv_labels(json_path, GT_path, img_text_path)
    print(f"INFO: CSV file created at {GT_path}")

    # Clear the img_resized directory ==================================================================
    GT_data = pd.read_csv(GT_path)
    id_list = GT_data.iloc[:, 0].tolist()

    print(f"INFO: Clearing the img_resized directory at {image_path}")
    cmpt = 0 
    for file in os.listdir(image_path):
        # remove the .jpeg extension
        if int(file[:-4]) not in id_list:
            try:
                os.remove(os.path.join(image_path, file))
                cmpt += 1
            except:
                print('Error while deleting file')
                pass

    print(cmpt, 'files deleted')
    print(f"INFO: img_resized directory cleared at {image_path}")

    # Transform images to tensors ======================================================================
    # check if the img_tens directory exists
    if not os.path.exists(image_path_tens):
        print(f"ERROR: The directory {image_path_tens} does not exist. Please create it before running this script.")
        return
    
    print(f"INFO: Transforming images to tensors in {image_path_tens}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cmpt = 0

    for img_name in os.listdir(image_path):
        cmpt += 1
        print(f'Processing image {cmpt}/{len(os.listdir(image_path))}')
        img_path = os.path.join(image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        new_img_path = os.path.join(image_path_tens, img_name)
        torch.save(img_tensor, new_img_path)

    print(f"INFO: Images transformed to tensors in {image_path_tens}")

if __name__ == '__main__':
    run_preprocessing()