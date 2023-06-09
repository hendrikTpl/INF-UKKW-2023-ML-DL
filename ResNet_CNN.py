import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 3)  # Adjust the output size to match your problem

    def forward(self, x):
        x = self.resnet(x)
        return x


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
    train_loss = running_loss / len(train_loader)
    return train_loss


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    return test_accuracy


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

    # Load the test dataset
    test_dataset_path = "C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/data/IndonesianStreetFood/test"
    test_dataset = torchvision.datasets.ImageFolder(test_dataset_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=2)

    # Create the model
    model = ResNetModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{10}, Training Loss: {train_loss}")

    # Save the model weights
    torch.save(model.state_dict(), "C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/core_model/model_weights.pth")

    # Test the model
    test_accuracy = test(model, test_loader)
    formatted_accuracy = "{:.2%}".format(test_accuracy)
    print(f"Test Accuracy: {formatted_accuracy}")


if __name__ == '__main__':
    main()
