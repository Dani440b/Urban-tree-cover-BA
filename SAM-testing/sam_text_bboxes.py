import os
import torch
from tqdm import tqdm
import warnings
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM

def data_transform(data):
    data = data - np.min(data)
    data = np.round(data * 255/np.max(data))
    data = data.astype(np.uint8)
    return np.array(data)

def train_patches(directory):
    transformed_data_list = []
    targets = []
    non_targets = []
    density = []
    transformed_data_list_eval = []
    targets_eval = []
    non_targets_eval = []
    density_eval = []
    
    # Initialize lists to store filenames
    train_filenames = []
    val_filenames = []

    print('Retrieving training data:')
    for i in tqdm(range(3566)):  # Range from 0 to 3565 for train samples
        filename = f"train_{i}.pt"
        train_filenames.append(filename)
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                data0 = torch.load(file_path)
                data1 = data0['aerial_20'].numpy()
                data1 = data1.transpose()
                transformed_data = data_transform(data1)
                transformed_data_list.append(transformed_data)
                
                target1 = data0['target'].numpy()
                targets.append(target1)
                
                nontargets1 = data0['target_mask'].numpy() - target1
                non_targets.append(nontargets1)

                density1 = data0['density'].numpy()
                density.append(density1)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print('Retrieving validation data:')
    for i in tqdm(range(788)):  # Range from 0 to 787 for val samples
        filename = f"val_{i}.pt"
        val_filenames.append(filename)
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                data0 = torch.load(file_path)
                data1 = data0['aerial_20'].numpy()
                data1 = data1.transpose()
                transformed_data = data_transform(data1)
                transformed_data_list_eval.append(transformed_data)
                
                target1 = data0['target'].numpy()
                targets_eval.append(target1)
                
                nontargets1 = data0['target_mask'].numpy() - target1
                non_targets_eval.append(nontargets1)

                density1 = data0['density'].numpy()
                density_eval.append(density1)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Return the filenames along with the data
    return (transformed_data_list, targets, non_targets, density, 
            transformed_data_list_eval, targets_eval, non_targets_eval, density_eval)

data, targets, non_targets, density, data_eval, targets_eval, non_targets_eval, density_eval = train_patches('patches')

print('Total train patches retrived:',len(data))
print('Total evaluation patches retrived:',len(data_eval))

image=data[2]

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

text_prompt = "trees"

def killl_train():
    # Suppress warning messages
    warnings.filterwarnings("ignore")

    # List to store boxes for each image
    boxes_list = []

    for i in range(len(data)):
        image = data[i]
        try:
            image_pil = Image.fromarray(image)

            model = LangSAM()
            masks, boxes, _, _ = model.predict(image_pil, text_prompt)
            print(boxes)

            if len(masks) == 0:
                print(f"No objects of the '{text_prompt}' prompt detected in the image.")
                # Append an empty box tensor if no masks are detected
                empty_box_tensor = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
                boxes_list.append(empty_box_tensor)
            else:
                # Convert boxes to a tensor if they are not already
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                # Save the boxes for this image
                boxes_list.append(boxes_tensor)

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error: {e}")

    # Save the list of tensors to a file
    torch.save(boxes_list, 'boxes_list_train.pt')

    # Now boxes_list contains the boxes tensors for each image
    return boxes_list

def killl_val():
    # Suppress warning messages
    warnings.filterwarnings("ignore")

    # List to store boxes for each image
    boxes_list2 = []

    for i in range(len(data_eval)):
        image = data_eval[i]
        try:
            image_pil = Image.fromarray(image)

            model = LangSAM()
            masks, boxes, _, _ = model.predict(image_pil, text_prompt)
            print(boxes)

            if len(masks) == 0:
                print(f"No objects of the '{text_prompt}' prompt detected in the image.")
                # Append an empty box tensor if no masks are detected
                empty_box_tensor = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
                boxes_list2.append(empty_box_tensor)
            else:
                # Convert boxes to a tensor if they are not already
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                # Save the boxes for this image
                boxes_list2.append(boxes_tensor)

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error: {e}")

    # Save the list of tensors to a file
    torch.save(boxes_list2, 'boxes_list_train.pt')

    # Now boxes_list contains the boxes tensors for each image
    return boxes_list2


lisst = killl_train()
lisst2 = killl_val()