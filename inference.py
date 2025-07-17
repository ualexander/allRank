#!/usr/bin/env python3
"""
Simple inference script for allRank model
"""

import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset_role
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.inference.inference_utils import rank_slates
from attr import asdict


def load_trained_model(model_path: str, config_path: str, test_ds):
    """Load trained model and config"""
    # Load config
    config = Config.from_json(config_path)

    # Get n_features from test dataset
    n_features = test_ds.shape[-1]

    # Create model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))

    # Load trained weights
    device = get_torch_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, config


def run_inference(model_path: str, config_path: str, test_data_path: str):
    """Run inference on test data"""
    print(f"Loading config from: {config_path}")
    config = Config.from_json(config_path)

    # Load test data first
    print(f"Loading test data from: {test_data_path}")
    import os
    import shutil

    temp_dir = "./temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(test_data_path, os.path.join(temp_dir, "test.txt"))

    test_ds = load_libsvm_dataset_role("test", temp_dir, config.data.slate_length)

    print(f"Loading model from: {model_path}")
    # Load model and config with test dataset
    model, config = load_trained_model(model_path, config_path, test_ds)

    # Run inference
    print("Running inference...")
    datasets = {"test": test_ds}
    ranked_results = rank_slates(datasets, model, config)

    # Print results
    test_X, test_y = ranked_results["test"]
    print(f"\nInference Results:")
    print(f"Number of queries: {test_X.shape[0]}")
    print(f"Number of documents per query: {test_X.shape[1]}")
    print(f"Number of features: {test_X.shape[2]}")

    # Show first query results
    print(f"\nFirst query ranking:")
    first_query_scores = test_y[0]
    print(f"Ranked relevance scores: {first_query_scores.tolist()}")

    return ranked_results


if __name__ == "__main__":
    # Paths
    model_path = "./test_run/results/test_run/model.pkl"
    config_path = "./test_run/results/test_run/used_config.json"
    test_data_path = "./test_data.txt"

    # Run inference
    results = run_inference(model_path, config_path, test_data_path)
    print("\nInference completed successfully!")
