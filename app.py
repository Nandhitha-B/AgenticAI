import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
# Using the path confirmed in the saved_model directory
model_path = "saved_model/brain_tumor_full_model.pth"
if not os.path.exists(model_path):
    # Fallback to current dir if needed, but we saw it in saved_model/
    model_path = "brain_tumor_full_model.pth"

try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the classes based on dataset folders
class_names = ["glioma", "meningioma", "pituitary"]

# Define the transformation
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img):
    if model is None:
        return "Model not found", {}
    
    # Preprocess image
    img = Image.fromarray(img).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    
    # Create dictionary of class name and confidence score
    results = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    # Get the class with highest confidence
    prediction = class_names[torch.argmax(probs).item()]
    
    return prediction, results

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Drag and drop or upload an MRI image"),
    outputs=[
        gr.Textbox(label="Predicted Tumor Class"),
        gr.Label(label="Confidence Scores", num_top_classes=3)
    ],
    title="MRI Tumor Classification",
    description="Upload an MRI scan to identify the type of tumor. The model can detect Glioma, Meningioma, and Pituitary tumors."
)

if __name__ == "__main__":
    demo.launch()
