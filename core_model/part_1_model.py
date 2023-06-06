# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch
import torch.nn as nn

# Membuat kelas model YourOwnCNN yang merupakan turunan dari nn.Module
class YourOwnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer konvolusi pertama
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer konvolusi kedua
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer fully connected pertama
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        
        # Layer fully connected kedua (output layer)
        self.fc2 = nn.Linear(64, 4)

    # Metode forward untuk melakukan forward pass pada model
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Mengecek model dengan membuat instance model dan mencetak informasi model
if __name__ =='__main__':
    model = YourOwnCNN()
    print(model)