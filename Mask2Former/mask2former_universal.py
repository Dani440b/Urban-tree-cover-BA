from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import numpy as np
from tqdm import tqdm

# Load Mask2Former trained on COCO instance segmentation dataset
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance"
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# Perform post-processing to get instance segmentation map
pred_instance_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
print(pred_instance_map.shape)

# Assuming your images are stored in a list called 'data'
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
    for i in tqdm(range(6)):  # Range from 0 to 3565 for train samples +1
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
    for i in tqdm(range(6)):  # Range from 0 to 787 for val samples +1
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


# Process each image in the data list
for i, img in enumerate(data):
    # Convert the image to PIL format and resize it to 224x224 (required input size for Mask2Former)
    img = Image.fromarray(img).resize((224, 224))

    # Convert the image to tensor format and normalize it
    img_t = F.to_tensor(img)
    img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Add an extra dimension at the beginning of the tensor (required for the model input)
    img_t = img_t.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(img_t, return_logits=True)

    # The output is a dictionary where output["pred_logits"] contains the raw logits
    logits = output["pred_logits"]

    # Apply sigmoid function to convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Now you can process the probabilities as needed    ...

    # Create a mask from the probabilities by thresholding
    mask = probabilities > 0.5

    # Overlay the mask on the original image
    img_np = np.array(img)
    mask_np = mask.squeeze().numpy()
    overlay = img_np.copy()
    overlay[mask_np] = [255, 0, 0]  # Red color for the mask

    # Save the overlay image in the 'image_examples' folder
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(os.path.join('image_examples', f'segmented_image_{i}.png'))

