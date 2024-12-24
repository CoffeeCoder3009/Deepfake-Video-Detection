import os
import cv2
from retinaface import RetinaFace

# Input and output directories
input_dir = "FF++_frame_output"
output_dir = "FF++_faces_RF"
output_dir_real = os.path.join(output_dir, "real_extracted")
output_dir_fake = os.path.join(output_dir, "fake_extracted")

# Create output directories
os.makedirs(output_dir_real, exist_ok=True)
os.makedirs(output_dir_fake, exist_ok=True)

# Function to process frames and extract faces
def process_frames(input_folder, output_folder, confidence_threshold=0.5):
    for video_folder in os.listdir(input_folder):
        video_input_path = os.path.join(input_folder, video_folder)
        
        # Create the corresponding output folder for this video
        video_output_path = os.path.join(output_folder, video_folder)
        os.makedirs(video_output_path, exist_ok=True)

        # Iterate through all images in the video folder
        for image_file in os.listdir(video_input_path):
            image_path = os.path.join(video_input_path, image_file)
            output_path = os.path.join(video_output_path, image_file)

            try:
                # Detect faces
                resp = RetinaFace.detect_faces(image_path)

                if isinstance(resp, dict):
                    for face_key, face_data in resp.items():
                        # Extract confidence score
                        score = face_data.get('score', 0)

                        # If the score is above the threshold, save the face
                        if score >= confidence_threshold:
                            # Get face bounding box
                            top_left = face_data['facial_area'][:2]
                            bottom_right = face_data['facial_area'][2:]

                            # Load the image
                            image = cv2.imread(image_path)

                            # Crop the face
                            face_crop = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                            # Save the cropped face
                            cv2.imwrite(output_path, face_crop)
                            print(f"Saved face from {image_path} to {output_path}")
                            break  # Only save one face per image
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Process frames from the 'real_extracted' and 'fake_extracted' directories in the input folder
process_frames(os.path.join(input_dir, "real_extracted"), output_dir_real, confidence_threshold=0.5)
process_frames(os.path.join(input_dir, "fake_extracted"), output_dir_fake, confidence_threshold=0.5)
