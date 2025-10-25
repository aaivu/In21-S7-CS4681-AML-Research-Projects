import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm

# Imports from our experiment
import config
from data_loader import WeatherDataset
from model import PatchTST_backbone

# Set seed for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, criterion, optimizer, train_loader, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
    for i, (batch_x, batch_y) in enumerate(loop):
        optimizer.zero_grad()

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        # The model expects shape [bs x nvars x seq_len]
        # Our loader provides [bs x seq_len x nvars], so we permute
        batch_x = batch_x.permute(0, 2, 1)

        outputs = model(batch_x)

        # The output is [bs x nvars x pred_len], we need to match it with batch_y
        # which is [bs x pred_len x nvars]. Let's permute output.
        outputs = outputs.permute(0, 2, 1)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0.0
    loop = tqdm(val_loader, desc='Validation', leave=True)
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loop):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x = batch_x.permute(0, 2, 1)

            outputs = model(batch_x)
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)

def main():
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = WeatherDataset(flag='train')
    val_dataset = WeatherDataset(flag='val')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model
    model = PatchTST_backbone(
        c_in=config.ENC_IN,  # Use ENC_IN from config (21 channels)
        context_window=config.SEQ_LEN,
        target_window=config.PRED_LEN,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.E_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        fc_dropout=config.FC_DROPOUT,
        head_dropout=config.HEAD_DROPOUT,
        individual=config.INDIVIDUAL,
        revin=config.REVIN,
        affine=config.AFFINE,
        subtract_last=config.SUBTRACT_LAST
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create results directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_pred{config.PRED_LEN}.pt')

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, epoch + 1)
        val_loss = validate(model, criterion, val_loader)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            epochs_no_improve = 0
            print(f'\t✓ Best model saved to {checkpoint_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config.PATIENCE:
                print(f'Early stopping after {config.PATIENCE} epochs with no improvement.')
                break

    print(f"\nTraining finished!")
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

if __name__ == '__main__':
    main()
