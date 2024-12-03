import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml
from models.model1 import SuperResolutionModel
from data.dataset import SuperResolutionDataset

def train_model(config):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SuperResolutionModel().to(device)
    
    # Create datasets and dataloaders
    train_dataset = SuperResolutionDataset(
        hr_dir=Path(config['data']['processed_data_path']) / 'train' / 'hr',
        lr_dir=Path(config['data']['processed_data_path']) / 'train' / 'lr'
    )
    
    val_dataset = SuperResolutionDataset(
        hr_dir=Path(config['data']['processed_data_path']) / 'val' / 'hr',
        lr_dir=Path(config['data']['processed_data_path']) / 'val' / 'lr'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Important for Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.L1Loss()
    
    # Set up tensorboard
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}') as pbar:
            for lr_imgs, hr_imgs in pbar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                optimizer.zero_grad()
                outputs = model(lr_imgs)
                
                # Calculate losses
                reconstruction_loss = criterion(outputs, hr_imgs)
                l1_reg_loss = model.l1_loss()
                loss = reconstruction_loss + l1_reg_loss
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_dir / 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

if __name__ == "__main__":
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    train_model(config)
