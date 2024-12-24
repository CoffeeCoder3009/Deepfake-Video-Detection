import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

# Define the same LSTM model as before
class DeepfakeLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_classes, num_layers=3, dropout=0.2, leaky_relu_slope=0.01):
        super(DeepfakeLSTM, self).__init__()
        
        # Define LSTM layer with multiple layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to output final classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        
    def forward(self, packed_input, lengths):
        # Unpack sequences
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Unpack the sequence output and pass through FC
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # We use the final hidden state of the LSTM for classification
        final_hidden_state = hn[-1]  # Get the last hidden state
        
        return final_hidden_state[-1]  # Return embedding (1x2048 sized)

# Function to extract embeddings from the model
def extract_embeddings(model, input_folder, output_folder, batch_size=4):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process 'real' and 'fake' video folders
    for label in ['real', 'fake']:
        label_folder = os.path.join(input_folder, f'{label}_extracted')
        output_label_folder = os.path.join(output_folder, label)
        os.makedirs(output_label_folder, exist_ok=True)
        
        # Iterate over subfolders (each subfolder is a video)
        for video_folder in os.listdir(label_folder):
            video_folder_path = os.path.join(label_folder, video_folder)
            if os.path.isdir(video_folder_path):
                print(f"Processing video folder: {video_folder_path}")

                # Load the sequence of .npy files (frames) from the video folder
                frames = []
                for frame_file in sorted(os.listdir(video_folder_path)):  # Sort to maintain temporal order
                    if frame_file.endswith('.npy'):
                        frame_path = os.path.join(video_folder_path, frame_file)
                        frame = np.load(frame_path)
                        frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        frames.append(frame_tensor)  # Add frame tensor

                # Stack frames to create a single tensor of shape (sequence_length, input_size)
                # If batch_size > 1, stack to create (batch_size, sequence_length, input_size)
                frames_tensor = torch.cat(frames, dim=0)  # shape: (sequence_length, input_size)
                frames_tensor = frames_tensor.unsqueeze(0)  # shape: (1, sequence_length, input_size)

                # Pad sequences to the max length in the batch
                lengths = torch.tensor([len(frames)])
                packed_frames = pack_padded_sequence(frames_tensor, lengths, batch_first=True, enforce_sorted=False)

                # Extract embedding using the model
                model.eval()
                with torch.no_grad():
                    embedding = model(packed_frames, lengths)
                
                # Save the embedding as a .npy file
                output_file_path = os.path.join(output_label_folder, f'{video_folder}_embedding.npy')
                np.save(output_file_path, embedding.numpy())  # Save as 1x2048 array

                print(f"Saved embedding for video: {video_folder}")

# Load pre-trained model (make sure to load the correct epoch/model)
model = DeepfakeLSTM(embedding_size=512, hidden_size=2048, num_classes=2)  # Same architecture as used during training

# Load the trained model's weights
model.load_state_dict(torch.load('C:/Users/c3ilab/Desktop/CV_Project/deepfake_lstm_epoch_9.pth'))  # Replace with your saved model's path

model.eval()  # Set the model to evaluation mode

# Extract embeddings
# extract_embeddings(model, input_folder, output_folder)
