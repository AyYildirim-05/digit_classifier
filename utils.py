from PIL import Image
from torchvision import transforms

def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image
