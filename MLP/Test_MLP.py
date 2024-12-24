# 1. Test the model after training

def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    
    # Disable gradient calculation for evaluation (it saves memory and computations)
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get the predicted class
            _, predicted = torch.max(outputs, 1)
            
            # Update correct and total counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Compute the test accuracy
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return accuracy, avg_loss

# 2. After training, evaluate the model on the test set

test_accuracy, test_loss = evaluate_model(model, test_loader, criterion)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

