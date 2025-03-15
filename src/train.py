import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import ResNet18
from utils.data_loader import get_data_loaders
from utils.metrics import MetricsTracker

def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = ResNet18().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize metrics tracker
    train_metrics = MetricsTracker()
    val_metrics = MetricsTracker()
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_metrics.reset()
        
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_metrics.update(outputs, labels, loss)
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        train_results = train_metrics.get_metrics()
        
        # Validation phase
        model.eval()
        val_metrics.reset()
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_metrics.update(outputs, labels, loss)
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
        
        val_results = val_metrics.get_metrics()
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_results['loss'], epoch)
        writer.add_scalar('Loss/val', val_results['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_results['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_results['accuracy'], epoch)
        
        print(f"\nTraining Loss: {train_results['loss']:.4f}, Accuracy: {train_results['accuracy']:.4f}")
        print(f"Validation Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
    
    writer.close()
    print("Training completed!")
    
    # Final evaluation
    print("\nEvaluating best model on test set...")
    best_model = ResNet18().to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = MetricsTracker()
    best_model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            test_metrics.update(outputs, labels, loss)
    
    test_results = test_metrics.get_metrics()
    print(f"\nTest Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")
    
    # Generate and save confusion matrix
    test_metrics.plot_confusion_matrix(
        save_path=os.path.join(args.log_dir, 'confusion_matrix.png')
    )
    test_metrics.print_classification_report()

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--log-dir', default='./logs', help='tensorboard log directory')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main() 