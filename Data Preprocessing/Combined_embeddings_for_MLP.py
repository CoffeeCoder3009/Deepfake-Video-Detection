import os
import numpy as np

# Define the folder paths
spatial_real_folder = 'C:/Users/c3ilab/Desktop/CV_Project/spatial_embeddings/real_extracted'
spatial_fake_folder = 'C:/Users/c3ilab/Desktop/CV_Project/spatial_embeddings/fake_extracted'
lstm_real_folder = 'C:/Users/c3ilab/Desktop/CV_Project/output_lstm_embeddings/real'
lstm_fake_folder = 'C:/Users/c3ilab/Desktop/CV_Project/output_lstm_embeddings/fake'
output_folder = 'C:/Users/c3ilab/Desktop/CV_Project/embeddings'  # Save embeddings here

# Create output folders for real and fake
os.makedirs(os.path.join(output_folder, 'real'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'fake'), exist_ok=True)

# Function to save temporal and spatial embeddings separately
def save_embeddings(video_name, spatial_folder, lstm_folder, output_folder, label):
    # Get paths to spatial and LSTM embeddings
    spatial_embedding_path = os.path.join(spatial_folder, video_name, 'spatial_embedding.npy')
    lstm_embedding_path = os.path.join(lstm_folder, f"{video_name}_embedding.npy")  # LSTM file is named with _embedding.npy suffix

    # Check if both embeddings exist
    if os.path.exists(spatial_embedding_path) and os.path.exists(lstm_embedding_path):
        print(f"Processing video: {video_name}")
        
        # Load the embeddings
        spatial_embedding = np.load(spatial_embedding_path)
        lstm_embedding = np.load(lstm_embedding_path)

        # Ensure the output folder exists for the current video
        video_output_folder = os.path.join(output_folder, label, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Save the embeddings separately
        np.save(os.path.join(video_output_folder, 'spatial_embedding.npy'), spatial_embedding)
        np.save(os.path.join(video_output_folder, 'temporal_embedding.npy'), lstm_embedding)
        
        print(f"Saved spatial and temporal embeddings for {video_name}")
    else:
        print(f"Missing embeddings for {video_name}, skipping.")

# Process all videos
def process_videos():
    # Get video names from the spatial real and fake folders
    real_video_names = os.listdir(spatial_real_folder)
    fake_video_names = os.listdir(spatial_fake_folder)

    # Process real videos
    for video_name in real_video_names:
        if os.path.isdir(os.path.join(spatial_real_folder, video_name)):  # Ensure it's a directory
            save_embeddings(video_name, spatial_real_folder, lstm_real_folder, output_folder, 'real')

    # Process fake videos
    for video_name in fake_video_names:
        if os.path.isdir(os.path.join(spatial_fake_folder, video_name)):  # Ensure it's a directory
            save_embeddings(video_name, spatial_fake_folder, lstm_fake_folder, output_folder, 'fake')

# Call the function to process the videos
process_videos()
