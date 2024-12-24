import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Dataset Class to Load Embeddings
class VideoEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_folder):
        self.embeddings_folder = embeddings_folder
        self.data = []
        self.labels = []

        # Check the 'real' and 'fake' folders
        for label_folder in ['real', 'fake']:
            label_path = os.path.join(embeddings_folder, label_folder)
            
            if os.path.isdir(label_path):
                # Iterate through each video folder inside 'real' and 'fake'
                for video_folder in os.listdir(label_path):
                    video_path = os.path.join(label_path, video_folder)
                    
                    if os.path.isdir(video_path):
                        spatial_embedding_path = os.path.join(video_path, 'spatial_embedding.npy')
                        temporal_embedding_path = os.path.join(video_path, 'temporal_embedding.npy')
                        
                        if os.path.exists(spatial_embedding_path) and os.path.exists(temporal_embedding_path):
                            # Load embeddings
                            spatial_embedding = np.load(spatial_embedding_path)
                            temporal_embedding = np.load(temporal_embedding_path)
                            
                            # Combine spatial and temporal embeddings into one vector
                            combined_embedding = np.concatenate((spatial_embedding, temporal_embedding), axis=0)
                            self.data.append(combined_embedding)
                            
                            # Label the video as real (0) or fake (1)
                            label = 0 if label_folder == 'real' else 1
                            self.labels.append(label)

        print(f"Loaded {len(self.data)} total embeddings.")  # Debugging line

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    # This method returns the number of samples in the dataset
    def __len__(self):
        return len(self.data)

    # This method is required by PyTorch to get the sample at index idx
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# 2. MLP Model Definition (2-layer perceptron)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # output raw logits for classification

# 3. Function to train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate train and validation accuracies
        train_accuracy = 100 * correct / total
        val_accuracy = 100 * correct_val / total_val
        
        # Save loss and accuracy for plotting
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print loss and accuracy
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save model after each epoch
        torch.save(model.state_dict(), f"mlp_epoch_{epoch+1}.pth")
        
        # Confusion matrix after each epoch
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=['Real', 'Fake'], columns=['Real', 'Fake'])
        print(f"Confusion Matrix after Epoch {epoch+1}:\n", cm_df)

    return train_losses, val_losses, train_accuracies, val_accuracies

# 4. Plotting the training and validation curves
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
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

# 5. Main function to run training and evaluation
if __name__ == "__main__":
    embeddings_folder = 'C:/Users/c3ilab/Desktop/CV_Project/embeddings'  # Path to embeddings folder
    dataset = VideoEmbeddingsDataset(embeddings_folder)

    # Split dataset into train, validation, and test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize the model
    input_size = len(dataset[0][0])  # Number of features (size of combined embedding)
    hidden_size = 512  # You can adjust this based on your problem
    model = MLP(input_size, hidden_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion, epochs=10
    )

    # Plot the results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)

    # Test the model
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    test_cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix on Test Data:\n", test_cm)
