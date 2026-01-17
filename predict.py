import torch

from model import DigitClassifier
from utils import preprocess_image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()

# Predict image
image = preprocess_image("digit.png").to(device)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1)

print("Predicted digit:", prediction.item())
