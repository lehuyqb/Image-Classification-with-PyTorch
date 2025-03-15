import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from .transforms import get_train_transforms, get_test_transforms

def get_data_loaders(data_dir='./data', batch_size=128, num_workers=4, validation_split=0.1):
    """
    Creates train, validation, and test data loaders for CIFAR-10.
    
    Args:
        data_dir (str): Directory to store/load the dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        validation_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get the transforms
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Download and load the training data
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Create validation split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Download and load the test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# CIFAR-10 classes
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
] 