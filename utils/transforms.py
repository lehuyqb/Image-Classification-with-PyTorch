import torch
from torchvision import transforms

# CIFAR-10 mean and std values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_train_transforms():
    """
    Returns the training data transforms with augmentation.
    
    The transforms include:
    - Random crop with padding
    - Random horizontal flip
    - Random rotation
    - Color jittering
    - Normalization
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_test_transforms():
    """
    Returns the test/validation data transforms.
    
    The transforms include only:
    - ToTensor conversion
    - Normalization
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_inference_transforms():
    """
    Returns transforms for single image inference.
    
    The transforms include:
    - Resize to 32x32
    - ToTensor conversion
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def inverse_normalize(tensor):
    """
    Inverse normalization for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
        std=[1/s for s in CIFAR10_STD]
    )
    return inv_normalize(tensor) 