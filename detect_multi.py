import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sys import argv

MODEL_PATH = 'best_wafer_model.pth'
IMAGE_PATH = argv[1]
CLASS_NAMES = ['center', 'cracks', 'edge', 'no_defects', 'open', 'others', 'particles', 'scratches'] 

AMBIGUITY_THRESHOLD = 20.0 

def predict_smart():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    top2_scores, top2_indices = torch.topk(probabilities, 2)
    
    score_1 = top2_scores[0].item() * 100  
    score_2 = top2_scores[1].item() * 100  
    
    class_1 = CLASS_NAMES[top2_indices[0].item()]
    class_2 = CLASS_NAMES[top2_indices[1].item()]

    print(f"Top Guess:    {class_1} ({score_1:.2f}%)")
    print(f"Second Guess: {class_2} ({score_2:.2f}%)")
    print("-" * 30)

    if score_2 > AMBIGUITY_THRESHOLD:
        print(f"Result: OTHER")
        print(f"(Reason: Model is confused between {class_1} and {class_2})")
    else:
        print(f"Result: {class_1.upper()}")

if __name__ == "__main__":
    predict_smart()
