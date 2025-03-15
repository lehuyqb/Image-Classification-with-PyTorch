import torch
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import ResNet18
from utils.transforms import get_inference_transforms, inverse_normalize
from utils.data_loader import CLASSES

def load_model(model_path, device):
    """
    Load the trained model
    
    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        
    Returns:
        model: Loaded model
    """
    model = ResNet18().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_path, device):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image_path (str): Path to the image
        device (torch.device): Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence_scores)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_inference_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return CLASSES[predicted.item()], probabilities.squeeze().cpu().numpy()

def plot_prediction(image_path, predicted_class, probabilities):
    """
    Plot the image with its prediction and confidence scores
    
    Args:
        image_path (str): Path to the image
        predicted_class (str): Predicted class name
        probabilities (numpy.ndarray): Confidence scores for each class
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot image
    image = Image.open(image_path).convert('RGB')
    ax1.imshow(image)
    ax1.set_title(f'Prediction: {predicted_class}')
    ax1.axis('off')
    
    # Plot confidence scores
    y_pos = np.arange(len(CLASSES))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASSES)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier Inference')
    parser.add_argument('--image-path', required=True, help='path to input image')
    parser.add_argument('--model-path', default='./checkpoints/best_model.pth',
                        help='path to trained model')
    parser.add_argument('--no-plot', action='store_true',
                        help='disable plotting of results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Make prediction
    try:
        predicted_class, probabilities = predict_image(model, args.image_path, device)
        print(f"\nPredicted class: {predicted_class}")
        print("\nConfidence scores:")
        for class_name, prob in zip(CLASSES, probabilities):
            print(f"{class_name}: {prob:.4f}")
        
        if not args.no_plot:
            plot_prediction(args.image_path, predicted_class, probabilities)
            
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == '__main__':
    main() 