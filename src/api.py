import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from models.resnet import ResNet18
from utils.transforms import get_inference_transforms
from utils.data_loader import CLASSES

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="A ResNet model for classifying images into 10 categories",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = get_inference_transforms()

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    model = ResNet18().to(device)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully on {device}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CIFAR-10 Image Classification API",
        "classes": CLASSES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make prediction on uploaded image
    
    Args:
        file: Image file to classify
        
    Returns:
        dict: Prediction results with class probabilities
    """
    # Read and preprocess image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Prepare response
    return {
        "class_name": CLASSES[predicted.item()],
        "confidence": float(confidence.item())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None} 