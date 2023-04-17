# /**
#  * @author Hendrik
#  * @email [hendrik.gian@gmail.com]
#  * @create date 2023-04-17 23:31:37
#  * @modify date 2023-04-17 23:31:37
#  * @desc [description]
#  */

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
"""
TODO: Build your first Custom CNN model
Input: 3x128x128
output: 3
CNN Architecture:

"""
# define class object model derived from nn.Module
class YourOwnCNN(nn.Module):
    # input
    # define convolutional layer 1
    # define pooling layer 1
    # define convolutional layer 2
    # define pooling layer 2
    # non-linear activation function
    # passing to forward function
    def __init__(self):
        super(YourOwnCNN).__init__()
        #define the desired architecture
        pass
        
        
    def forward(self, x):
        pass
    

## tester to check if the model is working
if __name__ =='__main__':
    model = SimpleCNN()
    print(model)
    x = torch.randn(1, 3, 128, 128)
    print(model(x))
    
        