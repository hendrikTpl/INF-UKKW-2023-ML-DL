import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from core_model.part_1_model import YourOwnCNN


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

    # Load the test dataset
    test_dataset_path = "C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/data/IndonesianStreetFood/test"
    test_dataset = torchvision.datasets.ImageFolder(test_dataset_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=2)

    # Create the model
    model = YourOwnCNN()

    # Load the saved model weights
    model.load_state_dict(torch.load("C:/Users/Asus/ML-2023/INF-UKKW-2023-ML-DL/core_model/model_weights.pth"))

    # Test the model
    test_accuracy = test(model, test_loader)
    formatted_accuracy = "{:.2%}".format(test_accuracy)  # Format test accuracy with two digits in front of the comma
    print(f"Test Accuracy: {formatted_accuracy}")


if __name__ == '__main__':
    main()
