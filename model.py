import torch
import torch.nn as nn

# Define the neural network model
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

# Input: 784 pixels (28x28 images)
# Hidden layer: 128 neurons
# Output: 10 classes (digits)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x