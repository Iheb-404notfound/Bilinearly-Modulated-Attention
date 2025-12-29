"""
Train a Vision Transformer on CIFAR-10 using BMA.

This script demonstrates how to use bilinearly modulated attention
for computer vision tasks and compare against standard attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from pathlib import Path
from tqdm import tqdm

from bma.pytorch import VisionTransformer


def get_dataloaders(batch_size=128, num_workers=4):
    """Create CIFAR-10 data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Creating ViT with {args.attention_type} attention...")
    model = VisionTransformer(
        image_size=32,
        patch_size=args.patch_size,
        n_classes=10,
        in_channels=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_type=args.attention_type
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = Path(f"best_vit_{args.attention_type}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'history': history
            }, save_path)
            print(f"Saved best model (Acc: {test_acc:.2f}%)")
    
    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--attention_type", type=str, default="bma",
                        choices=["bma", "standard", "gated"])
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
