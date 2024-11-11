# model.py
import torch
from torchvision import models, transforms
from PIL import Image

# Load the model with trained weights
def load_model():
    model = models.resnet50()  # Load a ResNet-50 model
    model.fc = torch.nn.Linear(model.fc.in_features, 200)  # Adjust for Tiny ImageNet classes
    model.load_state_dict(torch.load("resnet50_tiny_imagenet.pth", map_location="cpu"))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the uploaded image and run inference
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        predicted_class = predicted.item()  # Class index

    return predicted_class, torch.nn.functional.softmax(outputs, dim=1).max().item() * 100
