import torch
import torchvision
import torchmetrics
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
torch.use_deterministic_algorithms(False)
%env CUBLAS_WORKSPACE_CONFIG=:4096:8

import numpy as np
import torch
import matplotlib.pyplot as plt

import urllib.request
import os
from tqdm import tqdm

import re

#download the model

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
filepath = "sam_vit_h_4b8939.pth"

if not os.path.exists(filepath):
    file_size = int(urllib.request.urlopen(url).info().get("Content-Length", -1))

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, total=file_size, desc=filepath, ncols=80) as pbar:
        urllib.request.urlretrieve(url, filepath, reporthook=lambda b, bsize, t: pbar.update(bsize))
else:
    print("Checkpoint file already exists. Skipping download.")

#import and set the model

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# get and transform the data so that it matchges SAM

# Image shape: (N, H, W, C=3)
# Mask shape: (N, H, W)
# Values: [0, 255] (uint8)
import os
import torch
from tqdm import tqdm

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

data, targets, non_targets, density, data_eval, targets_eval, non_targets_eval, density_eval = train_patches('patches')

print('Total train patches retrived:',len(data))
print('Total evaluation patches retrived:',len(data_eval))

# defining helper functions

def find_pixel_target(target):
    my_array = target
    itemindex = np.where(my_array == 1)
    row_indices, col_indices = itemindex
    return row_indices, col_indices

# main function for random sampling from points and labels
# such that the remaining points and labels can be runn with SAM
# without requiring too much vram

def get_points_a_labels(targets, non_targets, num_to_remove=0, num_pos_remove=0, singular_target=False):
    positive_input_points = []
    positive_input_label = []
    negative_input_points = []
    negative_input_label = []
    
    if not singular_target:
        for i in range(len(targets)):
            pos_xs, pos_ys = find_pixel_target(targets)
            for j in range(len(pos_xs)):
                positive_input_points.append((pos_xs[j],pos_ys[j]))
                positive_input_label.append(1)
        
    for i in range(len(non_targets)):
        neg_xs, neg_ys = find_pixel_target(non_targets)
        for j in range(len(neg_xs)):
            negative_input_points.append((neg_xs[j],neg_ys[j]))
            negative_input_label.append(0)
    
    pos_labels = np.array(positive_input_label)
    pos_points = np.array(positive_input_points)
    neg_labels = np.array(negative_input_label)
    neg_points = np.array(negative_input_points)

    if (np.max(non_targets)==0) and (np.max(targets)==0):
            print('no points provided')
            return 

    if (num_to_remove > 0) and (num_pos_remove==0):
        num_to_remove = int(len(neg_points) * num_to_remove)
        indices = list(range(len(neg_points)))

        np.random.shuffle(indices)

        ne_po = [neg_points[i] for i in indices]

        remaining_neg_labels = neg_labels[num_to_remove:]
        remaining_neg_points = ne_po[num_to_remove:]

        if np.max(targets)==0:
            return np.array(remaining_neg_labels), np.array(remaining_neg_points)
        
        if np.max(non_targets)==0:
            return pos_labels, pos_points

        if singular_target:   
            if isinstance(targets, tuple) and len(targets) == 2:
                return np.concatenate(([1], remaining_neg_labels)), np.concatenate((([targets], remaining_neg_points)), axis=0)
            return np.concatenate(([1] * len(targets), remaining_neg_labels)), np.concatenate((([targets], remaining_neg_points)), axis=0)
        
        return np.concatenate((pos_labels, remaining_neg_labels)), np.concatenate((pos_points, remaining_neg_points))
    
    if (num_pos_remove > 0) and (num_to_remove > 0):
        num_pos_remove = int(len(pos_points) * num_pos_remove)
        indices_pos = list(range(len(pos_points)))
        num_to_remove = int(len(neg_points) * num_to_remove)
        indices = list(range(len(neg_points)))

        np.random.shuffle(indices_pos)
        np.random.shuffle(indices)

        pos_po = [pos_points[i] for i in indices_pos]
        ne_po = [neg_points[i] for i in indices]

        remaining_pos_labels = pos_labels[num_pos_remove:]
        remaining_pos_points = pos_po[num_pos_remove:]
        remaining_neg_labels = neg_labels[num_to_remove:]
        remaining_neg_points = ne_po[num_to_remove:]
        
        if np.max(targets)==0:
            return np.array(remaining_neg_labels), np.array(remaining_neg_points)
        
        if np.max(non_targets)==0:
            return np.array(remaining_pos_labels), np.array(remaining_pos_points)
        
        return np.concatenate((remaining_pos_labels, remaining_neg_labels)), np.concatenate((remaining_pos_points, remaining_neg_points))
    
    if np.max(targets)==0:
        return neg_labels, neg_points
    
    if np.max(non_targets)==0:
            return pos_labels, pos_points

    if singular_target:    
        if isinstance(targets, tuple) and len(targets) == 2:
                return np.concatenate(([1], remaining_neg_labels)), np.concatenate((([targets], neg_points)), axis=0)
        return np.concatenate(([1] * len(targets), neg_labels)), np.concatenate((targets, neg_points))

    return np.concatenate((pos_labels, neg_labels)), np.concatenate((pos_points, neg_points))

#set the images with targets and non_targets

threshold = 0
SAM_density_train = []
SAM_density_val = []
progress_bar = tqdm(range(len(data)))
zeroes_arr = torch.tensor(np.zeros((300, 300)))

print('Making softmax masks for training data')

for i in progress_bar:
    predictor.set_image(data[i])

    try:
        input_label, input_points = get_points_a_labels(targets[i], non_targets[i], num_to_remove=0.99995, num_pos_remove=0.999)
    except:
        SAM_density_train.append(zeroes_arr)
        progress_bar.set_description(f'mask {i} had no points and was skipped')
        continue

    masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=True,
            return_logits=True
        )
    
    mask_tensor = torch.from_numpy(masks[2])
    softmax_mask = torch.nn.functional.softmax(mask_tensor, dim=-1)
    mask = np.array(density[i]).transpose() == 0
    n_t = torch.tensor(np.array(non_targets[i]).transpose())
    softmax_mask -= n_t
    softmax_mask[softmax_mask < threshold] = 0
    softmax_mask[mask] = 0
    softmax_mask = np.transpose(softmax_mask)

    SAM_density_train.append(softmax_mask)

    progress_bar.set_description(f'mask {i+1} has score {scores[2]}')

progress_bar = tqdm(range(len(data_eval)))

print('Making softmax masks for validation data')

for i in progress_bar:
    predictor.set_image(data_eval[i])

    try:
        input_label, input_points = get_points_a_labels(targets_eval[i], non_targets_eval[i], num_to_remove=0.99995, num_pos_remove=0.999)
    except:
        SAM_density_val.append(zeroes_arr)
        progress_bar.set_description(f'mask {i} had no points and was skipped')
        continue

    masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=True,
            return_logits=True
        )
    
    mask_tensor = torch.from_numpy(masks[2])
    softmax_mask = torch.nn.functional.softmax(mask_tensor, dim=-1)
    mask = np.array(density_eval[i]).transpose() == 0
    n_t = torch.tensor(np.array(non_targets_eval[i]).transpose())
    softmax_mask -= n_t
    softmax_mask[softmax_mask < threshold] = 0
    softmax_mask[mask] = 0
    softmax_mask = np.transpose(softmax_mask)

    SAM_density_val.append(softmax_mask)

    progress_bar.set_description(f'mask {i+1} has score {scores[2]}')

# optimally we would now switch to the SAM auto mask generation,
# but it does return us a complete mask that we can use instead of density.
# SAM auto generate returns a dictionoary that looks like:
# dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
# therefore we simplly leave the density of these images for now

save_dir = 'patches_SAM'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_SAM_patches(directory, save_dir):
    for i in tqdm(range(3566)):  # Loop through indices for train samples
        if np.max(SAM_density_train[i])==0:
            continue
        filename = f"train_{i}.pt"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                patch = torch.load(file_path)
                new_patch = patch.copy()
                new_patch['density'] = SAM_density_train[i]
                save_file_path = os.path.join(save_dir, filename)
                torch.save(new_patch, save_file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    for i in tqdm(range(788)):  # Loop through indices for val samples
        if np.max(SAM_density_val[i])==0:
            continue
        filename = f"val_{i}.pt"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                patch = torch.load(file_path)
                new_patch = patch.copy()
                new_patch['density'] = SAM_density_val[i]
                save_file_path = os.path.join(save_dir, filename)
                torch.save(new_patch, save_file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

save_SAM_patches('patches', save_dir)







    