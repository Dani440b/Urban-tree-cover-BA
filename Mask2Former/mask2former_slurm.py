#importing model
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

#importing neccesary packages
from PIL import Image
import torch
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import cv2

#importing data from cached patches folder

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

    print('Retrieving training data:')
    for i in tqdm(range(3566)):  # Range from 0 to 3565 for train samples
        filename = f"train_{i}.pt"
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

    return (transformed_data_list, targets, non_targets, density, 
            transformed_data_list_eval, targets_eval, non_targets_eval, density_eval)

data, targets, non_targets, density, data_eval, targets_eval, non_targets_eval, density_eval = train_patches('~/new/citytrees_cache_no_test/patches_org')

print('Total train patches retrived:',len(data))
print('Total evaluation patches retrived:',len(data_eval))

def combine_masks_average_overlap(masks):
    # Create an array to count the number of non-zero entries for each pixel
    count_non_zero = np.zeros_like(masks[0], dtype=np.float32)

    # Create an array to sum the probabilities for each pixel
    sum_probabilities = np.zeros_like(masks[0], dtype=np.float32)

    for mask in masks:
        mask = np.array(mask)
        # Update the count of non-zero entries
        count_non_zero += (mask > 0)

        # Add the probabilities to the sum (this includes zeros)
        sum_probabilities += mask

    # To avoid division by zero, we set the count to 1 for pixels where it's zero
    count_non_zero[count_non_zero == 0] = 1

    # Compute the average only for overlapping areas
    combined_mask = sum_probabilities / count_non_zero

    return combined_mask

def make_masks(data, targets, non_targets):
    accumulator = []
    zero_tensor = torch.zeros((300, 300))
    for i, image_array in enumerate(data):
        maps = []

        image = Image.fromarray((image_array * 255).astype(np.uint8))
        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        masks_queries_logits = outputs.masks_queries_logits

        masks_queries_logits_resized = F.interpolate(masks_queries_logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        masks_queries_logits_sigmoid = torch.sigmoid(masks_queries_logits_resized)

        if np.max(targets[i])==0:
            accumulator.append(zero_tensor)
            continue

        for mask_index in range(masks_queries_logits_sigmoid.shape[1]):
            mask = masks_queries_logits_sigmoid[0, mask_index, :, :]
            mask[mask < 0.5] = 0

            binary_target = np.where(targets[i].transpose() == 1, 1, 0).astype(np.uint8)

            # note: use a smaller structuring element for faster operation (cv2 is fastest currently) 
            selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
            dilated_target = cv2.dilate(binary_target, selem)

            # Check if the entirety of the mask is within the dilated targets
            if np.array_equal(mask * dilated_target, mask):
                # Check if any part of the mask overlaps with the target
                if torch.sum(mask * targets[i].transpose()) > 0:
                    # Check if none of the mask overlaps with the non-target and then append
                    if torch.sum(mask * non_targets[i].transpose()) == 0:
                        maps.append(mask)

        combined_mask = combine_masks_average_overlap(maps)
        combined_mask = combined_mask.transpose()
        combined_mask = torch.from_numpy(combined_mask).to('cpu')
        accumulator.append(combined_mask)
    return accumulator

def save_mask2former_density(directory, save_dir):
    for i in tqdm(range(3566)):  # Loop through indices for train samples
        filename = f"train_{i}.pt"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                new_patch = torch.load(file_path)
                new_patch['density'] = m2f_density_train[i]
                save_file_path = os.path.join(save_dir, filename)
                torch.save(new_patch, save_file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    for i in tqdm(range(788)):  # Loop through indices for val samples
        filename = f"val_{i}.pt"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                new_patch = torch.load(file_path)
                new_patch['density'] = m2f_density_val[i]
                save_file_path = os.path.join(save_dir, filename)
                torch.save(new_patch, save_file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Use the coco pre-trained model
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

m2f_density_train = make_masks(data=data, targets=targets, non_targets=non_targets)
m2f_density_val = make_masks(data=data_eval, targets=targets_eval, non_targets=non_targets_eval)

save_dir = 'new/citytrees_cache_no_test/patches_m2f_coco'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_mask2former_density('new/citytrees_cache_no_test/patches_org', save_dir)


# Use the cityscapes pre-trained model
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

m2f_density_train = make_masks(data=data, targets=targets, non_targets=non_targets)
m2f_density_val = make_masks(data=data_eval, targets=targets_eval, non_targets=non_targets_eval)

save_dir = 'new/citytrees_cache_no_test/patches_m2f_cityscapes'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_mask2former_density('new/citytrees_cache_no_test/patches_org', save_dir)


# Use the ADE20K pre-trained model
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-panoptic")

m2f_density_train = make_masks(data=data, targets=targets, non_targets=non_targets)
m2f_density_val = make_masks(data=data_eval, targets=targets_eval, non_targets=non_targets_eval)

save_dir = 'new/citytrees_cache_no_test/patches_m2f_ade'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_mask2former_density('new/citytrees_cache_no_test/patches_org', save_dir)


# Use the mapillary-vistas pre-trained model
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")

m2f_density_train = make_masks(data=data, targets=targets, non_targets=non_targets)
m2f_density_val = make_masks(data=data_eval, targets=targets_eval, non_targets=non_targets_eval)

save_dir = 'new/citytrees_cache_no_test/patches_m2f_mapillary'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_mask2former_density('new/citytrees_cache_no_test/patches_org', save_dir)


