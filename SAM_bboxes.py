
import torch
import torchvision
import torchmetrics
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
torch.use_deterministic_algorithms(False)

import numpy as np
import matplotlib.pyplot as plt

import urllib.request
import os
from tqdm import tqdm
import pandas as pd

def show_mask(mask, ax, random_color=False, color_red=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color_red:
        color = np.array([255/255, 0/255, 0/255, 0.6])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=10):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=0.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=0.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
filepath = "sam_vit_h_4b8939.pth"

if not os.path.exists(filepath):
    file_size = int(urllib.request.urlopen(url).info().get("Content-Length", -1))

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, total=file_size, desc=filepath, ncols=80) as pbar:
        urllib.request.urlretrieve(url, filepath, reporthook=lambda b, bsize, t: pbar.update(bsize))
else:
    print("Checkpoint file already exists. Skipping download.")

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

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

def find_pixel_target(target):
    my_array = target
    itemindex = np.where(my_array == 1)
    row_indices, col_indices = itemindex
    return row_indices, col_indices

import numpy as np
from scipy.ndimage import label, center_of_mass

def find_clusters(data):
    labeled_array, num_clusters = label(data)

    centers = center_of_mass(data, labels=labeled_array, index=range(1, num_clusters+1))

    return centers

centers = find_clusters(targets[1])

for i, center in enumerate(centers, 1):
    print(f"Cluster {i} center: {center}")

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

def find_bounding_boxes(data, padding=15):
    labeled_array, num_clusters = label(data)
    bounding_boxes = []

    for i in range(1, num_clusters+1):
        positions = np.where(labeled_array == i)
        x_min, x_max = np.min(positions[1]) - padding, np.max(positions[1]) + padding
        y_min, y_max = np.min(positions[0]) - padding, np.max(positions[0]) + padding
        bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes


threshold = 0
SAM_density_train = []
SAM_density_val = []
progress_bar = tqdm(range(len(data)))
zeroes_arr = torch.tensor(np.zeros((300, 300)))


print('Making sigmoid masks for training data')

for i in progress_bar:
    predictor.set_image(data[i])

    if np.max(targets[i])==1:

        bboxes = find_bounding_boxes(targets[i].transpose())
        input_boxes = torch.tensor(bboxes, device=device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, data[i].shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            return_logits=True
            )
        
        combined_mask = np.max(np.stack(masks.cpu()), axis=0)
        combined_mask = np.reshape(combined_mask, (300,300))

        mask_tensor = torch.from_numpy(combined_mask)
        sigmoid_mask = torch.sigmoid(mask_tensor)
        mask = np.array(density[i]).transpose() == 0
        n_t = torch.tensor(np.array(non_targets[i]).transpose())
        sigmoid_mask -= n_t
        sigmoid_mask[sigmoid_mask < threshold] = 0
        sigmoid_mask[mask] = 0
        sigmoid_mask = np.transpose(sigmoid_mask)

        SAM_density_train.append(sigmoid_mask)

        progress_bar.set_description(f'mask {i+1} has score {scores[2]}')


    else:
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
        sigmoid_mask = torch.sigmoid(mask_tensor)
        mask = np.array(density[i]).transpose() == 0
        n_t = torch.tensor(np.array(non_targets[i]).transpose())
        sigmoid_mask -= n_t
        sigmoid_mask[sigmoid_mask < threshold] = 0
        sigmoid_mask[mask] = 0
        sigmoid_mask = np.transpose(sigmoid_mask)

        SAM_density_train.append(sigmoid_mask)

        progress_bar.set_description(f'mask {i+1} has score {scores[2]}')

progress_bar = tqdm(range(len(data_eval)))

print('Making sigmoid masks for validation data')

for i in progress_bar:
    predictor.set_image(data_eval[i])

    if np.max(targets_eval[i])==1:

        bboxes = find_bounding_boxes(targets_eval[i].transpose())
        input_boxes = torch.tensor(bboxes, device=device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, data_eval[i].shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            return_logits=True
            )
        
        combined_mask = np.max(np.stack(masks.cpu()), axis=0)
        combined_mask = np.reshape(combined_mask, (300,300))

        mask_tensor = torch.from_numpy(combined_mask)
        sigmoid_mask = torch.sigmoid(mask_tensor)
        mask = np.array(density_eval[i]).transpose() == 0
        n_t = torch.tensor(np.array(non_targets_eval[i]).transpose())
        sigmoid_mask -= n_t
        sigmoid_mask[sigmoid_mask < threshold] = 0
        sigmoid_mask[mask] = 0
        sigmoid_mask = np.transpose(sigmoid_mask)

        SAM_density_val.append(sigmoid_mask)

        progress_bar.set_description(f'mask {i+1} has score {scores[2]}')


    else:
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
        sigmoid_mask = torch.sigmoid(mask_tensor)
        mask = np.array(density_eval[i]).transpose() == 0
        n_t = torch.tensor(np.array(non_targets_eval[i]).transpose())
        sigmoid_mask -= n_t
        sigmoid_mask[sigmoid_mask < threshold] = 0
        sigmoid_mask[mask] = 0
        sigmoid_mask = np.transpose(sigmoid_mask)

        SAM_density_val.append(sigmoid_mask)

        progress_bar.set_description(f'mask {i+1} has score {scores[2]}')

save_dir = 'patches_SAM_boxes1'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_SAM_patches(directory, save_dir):
    for i in tqdm(range(3566)):  # Loop through indices for train samples
        filename = f"train_{i}.pt"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                new_patch = torch.load(file_path)
                if torch.max(SAM_density_train[i])==0:
                    pass
                else:
                    new_patch['density'] = SAM_density_train[i]
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
                if torch.max(SAM_density_val[i])==0:
                    pass
                else:
                    new_patch['density'] = SAM_density_val[i]
                save_file_path = os.path.join(save_dir, filename)
                torch.save(new_patch, save_file_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

save_SAM_patches('patches', save_dir)