import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = 'wafer_model.pth'
IMAGE_PATH = 'test_image.jpg'
CLASS_NAMES = ['center',  'cracks',  'edge',  'no_defects',  'open', 'others',  'particles',  'scratches'] 

device = torch.device("cuda")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    print(f"Prediction: {CLASS_NAMES[predicted_idx.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
