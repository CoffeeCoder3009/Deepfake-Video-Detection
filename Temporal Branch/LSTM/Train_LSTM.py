import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import numpy as np
import matplotlib.pyplot as plt

# Deepfake dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.root_dir = root_dir
        self.label_map = label_map
        self.video_files = []
        self.labels = []
        
        # Iterate over the main categories: 'train_lstm_real' and 'train_lstm_fake'
        for label_folder in os.listdir(root_dir):
            label_folder_path = os.path.join(root_dir, label_folder)
            
            if os.path.isdir(label_folder_path) and label_folder in self.label_map:
                for video_folder in os.listdir(label_folder_path):
                    video_path = os.path.join(label_folder_path, video_folder)
                    if os.path.isdir(video_path):
                        npy_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.npy')]
                        self.video_files.append(npy_files)
                        self.labels.append(self.label_map[label_folder])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        npy_files = self.video_files[idx]
        frames = [np.load(f) for f in npy_files]
        frames = np.array(frames)  # shape: (num_frames, embedding_size)
        
        label = self.labels[idx]
        
        # Convert frames to tensor
        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        
        return frames_tensor, label

# Pad sequences to the max length in the batch
def collate_fn(batch):
    frames, labels = zip(*batch)
    
    # Get the lengths of each sequence (number of frames in each video)
    lengths = torch.tensor([len(frame) for frame in frames])
    
    # Pad sequences to the max length in the batch
    padded_frames = pad_sequence([frame for frame in frames], batch_first=True, padding_value=0)
    
    # Pack the padded sequences
    packed_frames = pack_padded_sequence(padded_frames, lengths, batch_first=True, enforce_sorted=False)
    
    labels = torch.tensor(labels)
    
    return packed_frames, labels, lengths


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
        
        # Pass the final hidden state through the fully connected layer for classification
        output = self.fc(final_hidden_state)  # Classification layer
        output = self.leaky_relu(output)  # LeakyReLU activation
        
        # Return the final classification logits
        return output  # Logits for each class (batch_size x num_classes)


# Training loop
def train(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        for inputs, labels, lengths in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, lengths)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation loss and accuracy
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(100 * correct / total)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {100 * correct / total:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f"deepfake_lstm_epoch_{epoch+1}.pth")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            # Forward pass
            outputs = model(inputs, lengths)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Plotting function for accuracy and loss over epochs
def plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main code to run training
if __name__ == "__main__":
    # Define label mapping
    label_map = {'train_lstm_real': 0, 'train_lstm_fake': 1}

    # Define root directory for data
    root_dir = 'C:/Users/c3ilab/Desktop/CV_Project/train_LSTM'

    # Create dataset
    dataset = DeepfakeDataset(root_dir, label_map)

    # Split dataset into train, validation, and test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)

    # Initialize the model
    embedding_size = 512  # Size of each frame embedding
    hidden_size = 2048
    num_classes = 2  # Real vs Fake
    model = DeepfakeLSTM(embedding_size, hidden_size, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model, train_loader, val_loader, optimizer, criterion, epochs=10
    )

    # Plot the training results
    plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluate the model on test data
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
