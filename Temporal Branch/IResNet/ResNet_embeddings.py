import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn

# Load the fine-tuned ResNet model
resnet = models.resnet50(pretrained=True)

# Modify the fully connected layer (adjust output features if needed)
resnet.fc = torch.nn.Linear(2048, 512)  # Adjust based on your fine-tuning setup

# Load the state dict from the checkpoint
state_dict = torch.load("C:/Users/c3ilab/Desktop/CV_Project/classification_head_epoch_10.pth")

# Filter out the keys for the `fc` layer
pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

# Load the filtered weights into the model
resnet.load_state_dict(pretrained_dict, strict=False)

# Set the model to evaluation mode (for inference)
resnet.eval()

# Define the transformation to preprocess the images (same as you used for fine-tuning)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Input folder structure
input_dir = "C:/Users/c3ilab/Desktop/CV_Project/FF++_retina_face_output"  # Root folder
categories = ["fake_extracted", "real_extracted"]

# Output folder for saving embeddings
output_dir = "C:/Users/c3ilab/Desktop/CV_Project/temporal_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Process each category folder
for category in categories:
    category_path = os.path.join(input_dir, category)
    
    # Create a folder for each category (fake_extracted/real_extracted)
    output_category_path = os.path.join(output_dir, category)
    os.makedirs(output_category_path, exist_ok=True)  # Create category folder in output

    for folder_name in os.listdir(category_path):
        folder_path = os.path.join(category_path, folder_name)

        if os.path.isdir(folder_path):
            # Create a folder for this video in the category folder
            video_output_folder = os.path.join(output_category_path, folder_name)
            os.makedirs(video_output_folder, exist_ok=True)

            print(f"Processing folder: {folder_name} in category: {category}")

            # Process each frame (image) in the folder
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)

                try:
                    # Load the image and apply transformations
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

                    # Forward pass through the model to get the embedding
                    with torch.no_grad():
                        embedding = resnet(img_tensor)  # Extract feature

                    # Convert the embedding to a NumPy array and save it
                    embedding = embedding.cpu().numpy().flatten()  # Flatten to 1D array
                    output_file_path = os.path.join(video_output_folder, f"{image_file}_embedding.npy")
                    np.save(output_file_path, embedding)

                    print(f"Saved embedding for {image_file} to {output_file_path}")

                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")

print("Processing complete. Temporal embeddings saved to .npy files.")
