import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load Xception model (excluding top layers for embeddings)
base_model = Xception(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Input folder structure
input_dir = "C:/Users/c3ilab/Desktop/CV_Project/FF++_retina_face_output"  # Root folder
categories = ["fake_extracted", "real_extracted"]

# Output folder for saving embeddings
output_dir = "C:/Users/c3ilab/Desktop/CV_Project/spatial_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Process each category folder
for category in categories:
    category_path = os.path.join(input_dir, category)
    output_category_path = os.path.join(output_dir, category)
    os.makedirs(output_category_path, exist_ok=True)  # Create category folder in output
   
    for folder_name in os.listdir(category_path):
        folder_path = os.path.join(category_path, folder_name)
       
        if os.path.isdir(folder_path):
            # Create a folder for this video in the output directory
            video_output_folder = os.path.join(output_category_path, folder_name)
            os.makedirs(video_output_folder, exist_ok=True)
           
            embeddings_list = []
            print(f"Processing folder: {folder_name} in category: {category}")
           
            # Process each image in the folder
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
               
                try:
                    # Load and preprocess image
                    img = load_img(image_path, target_size=(299, 299))
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    img_array = np.expand_dims(img_array, axis=0)
                   
                    # Extract embedding
                    embedding = model.predict(img_array)
                    embeddings_list.append(embedding.flatten())
               
                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
           
            # Compute average embedding for the folder
            if embeddings_list:
                avg_embedding = np.mean(embeddings_list, axis=0)
                # Save average embedding to an .npy file inside the video folder
                output_file_path = os.path.join(video_output_folder, "spatial_embedding.npy")
                np.save(output_file_path, avg_embedding)
                print(f"Saved embedding for folder: {folder_name} to {output_file_path}")
            else:
                print(f"No valid images in folder: {folder_name}")

print("Processing complete. Spatial embeddings saved to .npy files.")
