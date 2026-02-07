import torch
import torch.nn as nn
import torchvision.models as models

INPUT_MODEL_PATH = 'wafer_model.pth'   
OUTPUT_ONNX_PATH = 'wafer.onnx'  
NUM_CLASSES = 7                  

def convert_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    try:
        model.load_state_dict(torch.load(INPUT_MODEL_PATH, map_location=device))
        print(f"Loaded weights from {INPUT_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_MODEL_PATH}")
        return

    model.eval() 
    model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("Exporting...")
    torch.onnx.export(
        model,                      
        dummy_input,                
        OUTPUT_ONNX_PATH,           
        export_params=True,         
        opset_version=18,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],    
        dynamic_axes={              
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Success! Model saved to {OUTPUT_ONNX_PATH}")

if __name__ == "__main__":
    convert_to_onnx()
