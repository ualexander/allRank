#!/usr/bin/env python3
"""
Debug script to trace exactly what's happening with the data
"""

import torch
import os
import shutil
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset_role
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.models import losses
from attr import asdict
from torch.utils.data import DataLoader


def debug_data_loading(test_data_path: str, config_path: str):
    """Debug what the data loader actually gives us"""
    print("=== DEBUGGING DATA LOADING ===")
    
    config = Config.from_json(config_path)
    
    # Load test data
    temp_dir = "./temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(test_data_path, os.path.join(temp_dir, "test.txt"))
    
    test_ds = load_libsvm_dataset_role("test", temp_dir, config.data.slate_length)
    
    print(f"Dataset shape: {test_ds.shape}")
    print(f"Dataset type: {type(test_ds)}")
    
    # Create DataLoader and examine first batch
    dataloader = DataLoader(test_ds, batch_size=1, num_workers=1, shuffle=False)
    
    for batch_idx, (xb, yb, indices) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"X shape: {xb.shape}")
        print(f"y shape: {yb.shape}")
        print(f"indices shape: {indices.shape}")
        
        print(f"\nTrue relevance labels (y): {yb[0].tolist()}")
        print(f"Indices: {indices[0].tolist()}")
        
        # Show first few features for each document
        for doc_idx in range(xb.shape[1]):
            print(f"Document {doc_idx} first 5 features: {xb[0, doc_idx, :5].tolist()}")
        
        break
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return test_ds


def debug_model_predictions(model_path: str, config_path: str, test_data_path: str):
    """Debug what the model actually predicts"""
    print("\n=== DEBUGGING MODEL PREDICTIONS ===")
    
    config = Config.from_json(config_path)
    
    # Load test data
    temp_dir = "./temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(test_data_path, os.path.join(temp_dir, "test.txt"))
    
    test_ds = load_libsvm_dataset_role("test", temp_dir, config.data.slate_length)
    n_features = test_ds.shape[-1]
    
    # Load model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    device = get_torch_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    
    # Get predictions
    dataloader = DataLoader(test_ds, batch_size=1, num_workers=1, shuffle=False)
    
    with torch.no_grad():
        for xb, yb, _ in dataloader:
            X = xb.type(torch.float32).to(device=device)
            y_true = yb.to(device=device)
            
            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            
            scores = model.score(X, mask, input_indices)
            
            print(f"True labels: {y_true[0].cpu().tolist()}")
            print(f"Predicted scores: {scores[0].cpu().tolist()}")
            print(f"Mask: {mask[0].cpu().tolist()}")
            
            # Show mapping
            for i in range(len(scores[0])):
                if not mask[0, i]:
                    print(f"Position {i}: True={y_true[0, i].item():.1f}, Predicted={scores[0, i].item():.4f}")
            
            break
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    model_path = "./test_run/results/test_run/model.pkl"
    config_path = "./test_run/results/test_run/used_config.json"
    test_data_path = "./test_data.txt"
    
    print("Raw test data:")
    with open(test_data_path, 'r') as f:
        for i, line in enumerate(f):
            print(f"Line {i}: {line.strip()}")
    
    debug_data_loading(test_data_path, config_path)
    debug_model_predictions(model_path, config_path, test_data_path)