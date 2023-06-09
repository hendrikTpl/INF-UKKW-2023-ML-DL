import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from core_model.part_1_model import YourOwnCNN


def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # Print loss every 100 mini-batches
            print(f"Epoch {epoch+1}/{10}, Mini-batch {i+1}, Loss: {running_loss / 100}")
            running_loss = 0.0
    train_loss = running_loss / len(train_loader)
    return train_loss


def main():
    # Set random seed for reproducibility
    torch.manual_seed(2023)

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training dataset
    train_dataset_path = "C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/data/IndonesianStreetFood/train"
    train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=2)

    # Create the model
    model = YourOwnCNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{10}, Training Loss: {train_loss}")

    # Save the model weights
    model_weights_path = "C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/core_model/model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved at: {model_weights_path}")

    print("Training completed.")


if __name__ == '__main__':
    main()
