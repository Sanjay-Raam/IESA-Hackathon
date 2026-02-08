# IESA-Hackathon

## Project Overview
This project uses **deep learning (ResNet-18)** with **transfer learning** to classify wafer defects in real time, achieving **very high** across multiple defect classes.

We selected ResNet-18 over deeper architectures (like ResNet-50) to balance classification accuracy with the lightweight efficiency required for embedded hardware
- Parameter Count: ~11 Million (Efficient).
- Transfer Learning: Initialized with ImageNet weights to leverage pre-learned texture/edge detection, fine-tuned on the wafer dataset.

The trained model is exportable to **ONNX** for integration with **edge runtimes** such as NXP eIQ.

## Key Features
- **High-Performance Accuracy:** The model achieves an aggregate accuracy, precision, and recall of 99% across eight distinct defect classes, ensuring reliable quality control
- **Optimized Lightweight Architecture:** Utilizes a ResNet-18 architecture optimized via Transfer Learning, striking a balance between high classification power and the lightweight efficiency required for embedded hardware.
- **Robust to Environmental Variations:** Implements strategic data augmentation, including geometric transformations (rotation/flipping) and photometric adjustments (brightness/contrast), to handle wafer rotation and lighting inconsistencies in fabrication plants
- **ONNX Runtime Integration:** Features a deployment workflow that exports the model to ONNX format
- **Real-Time Edge Inference:** Designed specifically for NXP i.MX RT series devices, the solution processes data locally to enable split-second decisions without the latency or bandwidth bottlenecks of cloud computing

## Model Details

- **Architecture:** ResNet-18 (~11M parameters)
- **Training Method:** Transfer Learning (ImageNet pretrained weights)
- **Total Training Images:** ~3,500+
- **Epochs:** 25
- **Optimizer:** SGD  
  - Learning Rate: `0.001`
  - Momentum: `0.9`
- **Loss Function:** Cross Entropy Loss
- **Final Layer:** Fully Connected layer replaced for custom defect classes
- **Input Resolution:** 224 × 224 images
---

## Classes
- Center
- Cracks
- Edge
- Open
- Particles
- Scratches
- Others
- No_defects
---

## Performance
| Metric      | Score |
|------------|-------|
| Accuracy   | 99%   |
| Precision  | 0.99  |
| Recall     | 0.99  |
| F1-Score   | 0.99  |

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## System Configuration
- **OS:** Arch Linux  
- **Python Version Manager:** pyenv  
- **Python Version:** 3.10.x
- **Cuda:** cu128

## Requirements
- **PyTorch:** `2.1.0`
- **TorchVision:** `0.14.0`
- **NumPy**
- **OpenCV**
- **ONNX**
> PyTorch was installed via pip using pyenv managed Python.

## Data Augmentation and Preprocessing
### Training Transformations
- Resize to `224x224`
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (±15°)
- ImageNet Normalization

### Validation Transformations
- Resize
- Tensor conversion
- Normalization only (no augmentation)

## Training Configuration
| Parameter  | Value |
|------------|-------|
| Batch Size | 8     |
| Epochs     | 25    |
| Optimizer  | SGD   |
| Learning Rate| 0.001|
| Momentum   | 0.9   |
| Loss Function | Cross Entropy Loss |
| LR Scheduler | StepLR |
| Device     | CUDA   |
