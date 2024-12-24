import os
import cv2
from retinaface import RetinaFace
import numpy as np
from tensorflow.keras.applications import Xception  # Pretrained Xception model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
import torch
import torch.nn as nn
import torch.optim as optim
import insightface
from torch.utils.data import DataLoader
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Paths to the input and output directories
input_dir = "FF++"
output_dir = "FF++_frame_output"
output_dir_real = os.path.join(output_dir, "real_extracted")
output_dir_fake = os.path.join(output_dir, "fake_extracted")

# Ensure the main output directory and subdirectories exist
os.makedirs(output_dir_real, exist_ok=True)
os.makedirs(output_dir_fake, exist_ok=True)

# Function to extract frames from videos
def extract_frames(video_path, output_folder, frame_interval=25):
    # Create a folder for the current video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    # Iterate through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count % frame_interval == 0:
            # Save the current frame
            frame_filename = os.path.join(video_output_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path} into {video_output_path}")

# Process videos in "real" and "fake" folders
for category in ["real", "fake"]:
    input_folder = os.path.join(input_dir, category)
    output_folder = output_dir_real if category == "real" else output_dir_fake
    
    # Iterate through all files in the category folder
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if os.path.isfile(video_path):  # Check if it's a file
            extract_frames(video_path, output_folder, frame_interval=25)
