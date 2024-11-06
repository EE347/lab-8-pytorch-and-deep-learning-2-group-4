import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torch import nn

if __name__ == '__main__':

    # Choose the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Loss lists
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []

    # Saving parameters
    best_train_loss_ce = 1e9
    best_train_loss_nll = 1e9

    # CrossEntropyLoss and NLLLoss criteria
    criterion_ce = nn.CrossEntropyLoss()
    criterion_nll = nn.NLLLoss()

    # Function to train and evaluate using a given loss function
    def train_and_evaluate(loss_fn, criterion):
        # Epoch Loop
        train_loss = 0
        test_loss = 0
        correct = 0
        total = 0
        model.train()

        # Train the model
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = images.permute(0, 3, 1, 2).to(device)  # Change to [batch, channels, height, width]
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            if loss_fn == 'ce':  # CrossEntropyLoss
                loss = criterion(outputs, labels)
            elif loss_fn == 'nll':  # NLLLoss
                log_probs = torch.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            train_loss += loss.item()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(testloader, total=len(testloader), leave=False):
                images = images.permute(0, 3, 1, 2).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute the loss
                if loss_fn == 'ce':  # CrossEntropyLoss
                    loss = criterion(outputs, labels)
                elif loss_fn == 'nll':  # NLLLoss
                    log_probs = torch.log_softmax(outputs, dim=1)
                    loss = criterion(log_probs, labels)

                # Accumulate the test loss
                test_loss += loss.item()

                # Get the predicted class from the maximum value in the output-list of class scores
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)

                # Accumulate the number of correct classifications
                correct += (predicted == labels).sum().item()

        return train_loss, test_loss, correct, total

    # Training and evaluation for CrossEntropyLoss
    print("Training with CrossEntropyLoss...")
    train_loss_ce, test_loss_ce, correct_ce, total_ce = train_and_evaluate('ce', criterion_ce)
    train_accuracy_ce = correct_ce / total_ce

    # Training and evaluation for NLLLoss
    print("Training with NLLLoss...")
    train_loss_nll, test_loss_nll, correct_nll, total_nll = train_and_evaluate('nll', criterion_nll)
    train_accuracy_nll = correct_nll / total_nll

    # Print the results for both loss functions
    print(f"CrossEntropyLoss Results: Train Loss: {train_loss_ce / len(trainloader):.4f}, "
          f"Test Loss: {test_loss_ce / len(testloader):.4f}, Test Accuracy: {train_accuracy_ce:.4f}")
    print(f"NLLLoss Results: Train Loss: {train_loss_nll / len(trainloader):.4f}, "
          f"Test Loss: {test_loss_nll / len(testloader):.4f}, Test Accuracy: {train_accuracy_nll:.4f}")

    # Store the results
    train_losses_ce.append(train_loss_ce / len(trainloader))
    test_losses_ce.append(test_loss_ce / len(testloader))
    train_losses_nll.append(train_loss_nll / len(trainloader))
    test_losses_nll.append(test_loss_nll / len(testloader))

    # Save the model with the best performance (based on loss)
    if train_loss_ce < best_train_loss_ce:
        best_train_loss_ce = train_loss_ce
        torch.save(model.state_dict(), 'lab8/best_model_ce.pth')

    if train_loss_nll < best_train_loss_nll:
        best_train_loss_nll = train_loss_nll
        torch.save(model.state_dict(), 'lab8/best_model_nll.pth')

    # Save the model
    torch.save(model.state_dict(), 'lab8/current_model.pth')

    # Create the loss plot for both
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_ce, label='Train Loss (CrossEntropyLoss)')
    plt.plot(test_losses_ce, label='Test Loss (CrossEntropyLoss)')
    plt.plot(train_losses_nll, label='Train Loss (NLLLoss)')
    plt.plot(test_losses_nll, label='Test Loss (NLLLoss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lab8/task5_loss_plot.png')