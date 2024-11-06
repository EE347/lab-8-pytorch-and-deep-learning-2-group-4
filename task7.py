import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Saving parameters
    best_train_loss = 1e9

    # Loss lists
    train_losses = []
    test_losses = []

    # Initialize confusion matrix containers
    all_preds = []
    all_labels = []

    # Set up live plotting
    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # Epoch Loop
    for epoch in range(1, 100):

        # Start timer
        t = time.time_ns()

        # Train the model
        model.train()
        train_loss = 0

        # Batch Loop
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            train_loss = train_loss + loss.item()

        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # Reset confusion matrix containers
        epoch_preds = []
        epoch_labels = []

        # Batch Loop for testing
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Accumulate the loss
            test_loss = test_loss + loss.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

            # Save predictions and true labels for confusion matrix
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        # Update the overall confusion matrix
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        # Compute the confusion matrix using sklearn
        cm = confusion_matrix(all_labels, all_preds)

        # Clear the previous plot before updating with the new confusion matrix
        ax.clear()

        # Plot the confusion matrix
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Epoch {epoch} Confusion Matrix')
        plt.colorbar(ax.imshow(cm))

        # Labeling the axes and adding annotations
        tick_marks = np.arange(len(np.unique(all_labels)))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(np.unique(all_labels))
        ax.set_yticklabels(np.unique(all_labels))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Annotate each cell in the confusion matrix with its numeric value
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, f'{cm[i, j]}', horizontalalignment="center", color="black")

        # Draw the live confusion matrix
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

        # Print the epoch statistics
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Update loss lists
        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        # Update the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'lab8/best_model.pth')

        # Save the model
        torch.save(model.state_dict(), 'lab8/current_model.pth')

        # Create the loss plot (you can still save this to track loss)
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task2_loss_plot.png')

    # Close the interactive plot window at the end of training
    plt.ioff()
    plt.show()